import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
from dotenv import load_dotenv
from keitou_data import keitou_reference_text

load_dotenv()

app = Flask(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# gemini-2.5-flashはデフォルトで思考(thinking)機能がONになっており、
# 単純な文章生成でも内部で推論トークンを消費して数秒〜十数秒遅くなる。
# スタイル診断・説明文生成は思考不要なタスクなのでOFFにして高速化する。
FAST_CONFIG = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=0)
)

DISCLAIMER = """✨ご購入前に、ぜひ読んでいただきたいこと✨

古着という特性上、あらかじめ知っておいていただきたいことをまとめました。

◆サイズ表記だけでなく、必ず実寸サイズをご確認ください
古着は同じ表記サイズでも、ブランドや年代によってサイズ感が異なります。

◆写真は加工せず、スマホで撮影したそのままの実物です
光の当たり方で多少印象が変わる場合があります。

◆「目立った傷・汚れなし」の基準について
多少の使用感（軽い毛羽立ちやごく小さな汚れなど）がパッと見て気にならない範囲を指します。
匂いや細かなダメージなど見落としてしまう場合もあるので、気になる点はお気軽にコメントください☺

◎発送の際はコンパクトに畳んでお送りします。到着後、軽くしわを伸ばしていただくと綺麗に着ていただけます☘"""

FOLLOW_DISCOUNT = """✨最後まで読んでくださってありがとうございます✨
当店をフォローしていただいた方には、ちょっとしたお値引きをさせていただいております☺
よろしければ他のお品物も覗いてみてください☘"""

# mercari-kpiでの分析（2026-09-30、全期間21件）でわかった季節・ジャンル別の平均利益の傾向。
# 母数が少ないため断定はせず、あくまで「参考情報」として扱う。数字が更新されたらここも直す。
SALES_INSIGHT = """【参考：これまでの販売データからわかっている傾向（母数21件、目安程度に）】
季節別の平均利益：秋冬¥1,842 > 春夏¥931 ≈ オールシーズン¥900（秋冬が他の季節の約2倍）
ジャンル別の平均利益：アウター¥2,299 > ワンピース¥1,320 > トップス¥1,281 > ボトムス¥902"""


def current_season_label():
    """今日の月から「秋冬」「春夏」を判定する（10〜3月は秋冬、4〜9月は春夏）"""
    return "秋冬" if date.today().month in (10, 11, 12, 1, 2, 3) else "春夏"


# 実際に閲覧数が多かった／早く売れた商品のタイトル・説明文の実例。
# 売れた商品が出るたびにここへ追加していくと、生成の精度が上がっていく。
# {"title": "実際のタイトル", "description": "実際の説明文（本文のみでOK、注意書き等は除く）"} の形で追加する。
GOOD_EXAMPLES = []


def good_examples_text():
    """プロンプトに差し込む用の、過去の好例テキスト。例が無ければ空文字を返す"""
    if not GOOD_EXAMPLES:
        return ""
    blocks = [
        f"例{i}\nタイトル：{ex['title']}\n説明文：{ex['description']}"
        for i, ex in enumerate(GOOD_EXAMPLES, 1)
    ]
    return (
        "【実際に反応が良かった過去の例】\n"
        "検索ワードの選び方・魅力の伝え方の雰囲気を参考にすること。内容をそのままコピーしないこと。\n"
        + "\n\n".join(blocks)
    )


def format_measurements(measurements):
    # 「その他」カテゴリでの自由入力時、数値の直後に付いた cm 表記（全角/半角・大文字小文字問わず）が
    # 次のラベルに巻き込まれて「✿cm身幅：約50cm」のように重複表示されるのを防ぐため、先に取り除く
    cleaned = re.sub(r'(\d)\s*(?:cm|ｃｍ|CM|ＣＭ|㎝)', r'\1', measurements)
    pairs = re.findall(r'([^\d\s：:]+)[：:]?(\d+)', cleaned)
    if not pairs:
        return measurements
    return '\n'.join(f'✿{label}：約{num}cm' for label, num in pairs)


def extract_section(text, section):
    match = re.search(rf'【{re.escape(section)}】\s*([\s\S]*?)(?=\n【|$)', text)
    if not match:
        return ''
    return re.sub(r'^\(.*?\)\s*', '', match.group(1).strip(), flags=re.S).strip()


def extract_title_candidates(text):
    """系統が決めづらい商品向け：「候補1：（系統名）タイトル」形式の行を複数抽出する"""
    section = extract_section(text, 'タイトル')
    matches = re.findall(r'候補\d+[：:]\s*(?:[（(](.*?)[）)])?\s*(.+)', section)
    if not matches:
        return [{"style": None, "title": enforce_title_length(section)}]
    return [
        {"style": style.strip() if style else None, "title": enforce_title_length(title.strip())}
        for style, title in matches
    ]


def enforce_title_length(title, limit=40):
    if len(title) <= limit:
        return title
    words = title.split()
    while len(words) > 1 and len(' '.join(words)) > limit:
        words.pop()
    result = ' '.join(words)
    return result[:limit]

def enforce_description_length(appeal, hashtags, detail_text, limit=1000):
    def build(a):
        return '\n\n'.join([a, detail_text, DISCLAIMER, FOLLOW_DISCOUNT, hashtags])

    description = build(appeal)
    if len(description) <= limit:
        return description

    lines = appeal.split('\n')
    while len(lines) > 1 and len(build('\n'.join(lines))) > limit:
        lines.pop()
    appeal = '\n'.join(lines)
    description = build(appeal)

    if len(description) > limit:
        overage = len(description) - limit
        appeal = appeal[:-overage] if overage < len(appeal) else ''
        description = build(appeal)

    return description

def analyze_style(query):
    prompt = f"""
あなたはファッションの専門家です。
以下の入力に対して、古着・メルカリ出品に役立つ情報を教えてください。

入力：{query}

【当店で使っている系統一覧】（スタイル名は、できる限りこの中から選んでください。当てはまるものが無い場合のみ新しい名前にしてください）
{keitou_reference_text()}

以下の形式で答えてください。

【スタイル名】
（該当するファッションスタイルの名前。複数ある場合はカンマ区切りで）

【説明】
（そのスタイルの特徴を2〜3文で。初心者にもわかりやすく）

【メルカリで使えるキーワード】
（検索されやすいハッシュタグ向けキーワードを5〜8個。カンマ区切りで）

【こんな商品に使える】
（どんなアイテム・色・素材に合うか1〜2文で）
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt, config=FAST_CONFIG
    )
    return response.text

def generate_description(info):
    style_line = f"スタイル・雰囲気：{info['style']}" if info.get('style') else ''
    season = current_season_label()
    examples = good_examples_text()
    examples_block = f"\n{examples}\n" if examples else ""
    uncertain_style = bool(info.get('uncertain_style'))

    # 系統一覧・販売傾向・商品情報など、タイトル用プロンプトと説明文用プロンプトの両方で必要な部分。
    # 2本に分けて並行実行する分、入力トークンは2重になるが、生成時間を支配するのは出力の長さなので
    # 入力が多少増えてもレイテンシへの影響は小さい。
    context_block = f"""
あなたはメルカリ出品のプロです。

【当店で使っている系統一覧】
{keitou_reference_text()}

{SALES_INSIGHT}
今は「{season}」の時期です。商品がこの季節物として自然に当てはまる場合や、ジャンルがアウター・ワンピースなど単価が高い傾向にある場合は、保温性・重ね着のしやすさ・季節感など単価が乗りやすい訴求ポイントを意識してください。ただし、商品に関係ない季節・ジャンルを無理にこじつけたり、事実にない特徴を書き加えたりしないこと。
{examples_block}
【商品情報】
ブランド名：{info['brand']}
アイテム：{info['item']}
サイズ（表記）：{info['size']}
実寸：{info['measurements']}
色：{info['color']}
状態：{info['condition']}
素材：{info.get('material') or '不明'}
{style_line}
"""

    if uncertain_style:
        title_instruction = (
            "（この商品は系統が1つに決めづらいため、上の【当店で使っている系統一覧】から異なる系統を3つ選び、"
            "それぞれを軸にしたタイトル候補を3つ作ること。各候補は単独で40文字ちょうどになるまで使い切り、"
            "文章にせず検索されやすい単語を並べる形にする。単語と単語の間は必ず半角スペースで区切り、続けて書かないこと。"
            "アイテム名、サイズ、色、ブランド名、素材、系統・デザインの特徴を優先度が高い順に並べ、優先度が低い単語から"
            "先に削って調整すること。系統ワードは各候補1個までにする。\n"
            "必ず以下の形式で、3行だけ出力すること（説明文などはここに書かない）：\n"
            "候補1：（系統名）単語 単語 単語 単語（スペース区切りのタイトル本体）\n"
            "候補2：（系統名）単語 単語 単語 単語（スペース区切りのタイトル本体）\n"
            "候補3：（系統名）単語 単語 単語 単語（スペース区切りのタイトル本体））"
        )
        appeal_note = "この商品は系統が複数候補あるため、特定の系統の読み手に限定しすぎず、どの候補タイトルで見た人が読んでも違和感のない書き方にすること。"

        title_prompt = f"""{context_block}
【出力形式】必ず以下の形式で出力してください。

【タイトル】
{title_instruction}
"""
        desc_prompt = f"""{context_block}
【出力形式】必ず以下の形式で出力してください。

【説明文】
（商品の魅力を3つのポイントに絞り、1ポイント1行、各行の先頭に「✅」を付けて箇条書きにすること。長い文章は読まれないため、1行は20〜30文字程度の短い一文にまとめる。ポイントはデザインの特徴・素材感・着こなし方・季節感などから、その商品に合うものを3つ選ぶ。着こなし方を提案する行は、その商品自体の系統・雰囲気に自然に合うものにすること（特定のコンセプトに無理に寄せない）。{appeal_note}色・素材・状態など商品自体の事実は正確に書き、誇張・変更しないこと。見出し・実寸・注意書きは書かない。行頭の✅以外の記号（✨・☺・☘など）は文中で使わず、ごちゃごちゃしないようにする）

【ハッシュタグ】
（5〜8個。メルカリで検索されやすいものを選ぶ）
"""
        try:
            # タイトル3候補と説明文を1本のプロンプトで順番に書かせると出力量が多く遅いため、
            # 2本の呼び出しに分けて並行実行し、待ち時間を「長い方の呼び出し時間」まで短縮する。
            with ThreadPoolExecutor(max_workers=2) as executor:
                title_future = executor.submit(
                    client.models.generate_content,
                    model="gemini-2.5-flash", contents=title_prompt, config=FAST_CONFIG
                )
                desc_future = executor.submit(
                    client.models.generate_content,
                    model="gemini-2.5-flash", contents=desc_prompt, config=FAST_CONFIG
                )
                title_text = title_future.result().text
                desc_text = desc_future.result().text
        except Exception as e:
            print("Gemini APIエラー:", repr(e))
            if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                return {"error": "只今アクセスが集中しています。1分ほど待ってからもう一度お試しください。"}
            return {"error": "生成に失敗しました。もう一度お試しください。"}

        titles = extract_title_candidates(title_text)
        appeal = extract_section(desc_text, '説明文')
        hashtags = extract_section(desc_text, 'ハッシュタグ')
    else:
        title_instruction = (
            "（40文字ちょうどになるまで使い切ること。文章にせず、検索されやすい単語を並べる形にする。アイテム名、サイズ、色、"
            "ブランド名（英語表記とカタカナ表記の両方が一般的なら両方）、素材、系統・デザインの特徴を、優先度が高い順に並べて"
            "40文字に収まるだけ詰め込む。優先度が低い単語（素材・系統・デザインの特徴）から先に削って調整すること。ただし"
            "系統・雰囲気を表す単語は、複数思いついても合計1〜2個までに絞ること。使う場合は上の【当店で使っている系統一覧】の"
            "中から商品に最も近いものを選び、一覧に無い独自の雰囲気ワードを何個も並べないこと）"
        )

        prompt = f"""{context_block}
【出力形式】必ず以下の形式で出力してください。

【タイトル】
{title_instruction}

【説明文】
（商品の魅力を3つのポイントに絞り、1ポイント1行、各行の先頭に「✅」を付けて箇条書きにすること。長い文章は読まれないため、1行は20〜30文字程度の短い一文にまとめる。ポイントはデザインの特徴・素材感・着こなし方・季節感などから、その商品に合うものを3つ選ぶ。着こなし方を提案する行は、その商品自体の系統・雰囲気に自然に合うものにすること（特定のコンセプトに無理に寄せない）。色・素材・状態など商品自体の事実は正確に書き、誇張・変更しないこと。見出し・実寸・注意書きは書かない。行頭の✅以外の記号（✨・☺・☘など）は文中で使わず、ごちゃごちゃしないようにする）

【ハッシュタグ】
（5〜8個。メルカリで検索されやすいものを選ぶ）
"""
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt, config=FAST_CONFIG
            )
            text = response.text
        except Exception as e:
            print("Gemini APIエラー:", repr(e))
            if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                return {"error": "只今アクセスが集中しています。1分ほど待ってからもう一度お試しください。"}
            return {"error": "生成に失敗しました。もう一度お試しください。"}

        appeal = extract_section(text, '説明文')
        hashtags = extract_section(text, 'ハッシュタグ')

    detail_parts = [
        f"【ブランド】\n{info['brand']}",
        f"【状態】\n{info['condition']}",
    ]

    material = info.get('material')
    if material:
        detail_parts.append(f"【素材】\n{material}")
    else:
        detail_parts.append(
            "【素材】\n"
            "タグの摩耗・欠損により素材表記の確認ができないため、記載を省略しております。\n"
            "ご不明点はコメントにてお問い合わせください。"
        )

    detail_parts.append(
        "【サイズ】\n"
        f"表記：{info['size']}\n"
        "実寸\n"
        f"{format_measurements(info['measurements'])}\n"
        "※素人採寸のため、多少の誤差はご容赦ください。"
    )

    description = enforce_description_length(appeal, hashtags, '\n\n'.join(detail_parts))

    if uncertain_style:
        return {"titles": titles, "description": description, "hashtags": hashtags}
    return {"title": enforce_title_length(extract_section(text, 'タイトル')), "description": description, "hashtags": hashtags}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    info = request.json
    print("受信データ:", info)
    result = generate_description(info)
    if 'error' in result:
        return jsonify(result), 502
    return jsonify(result)

@app.route("/style", methods=["POST"])
def style():
    query = request.json.get("query", "")
    result = analyze_style(query)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", threaded=True)
