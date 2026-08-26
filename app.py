import os
import re
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types
from dotenv import load_dotenv

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

BRAND_CONCEPT = """海外ガールのような、程よい抜け感のある着こなしを大切にしています。
アイテムで着飾るというより、色や小物の使い方、そして着る人自身の雰囲気で魅せるスタイル。
デニムにTシャツのようなシンプルな組み合わせでも、個性が光る着こなしを提案します。"""


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
    prompt = f"""
あなたはメルカリ出品のプロです。
以下の商品情報をもとに、タイトル・説明文・ハッシュタグを作成してください。

【当店のコンセプト】
{BRAND_CONCEPT}

【商品情報】
ブランド名：{info['brand']}
アイテム：{info['item']}
サイズ（表記）：{info['size']}
実寸：{info['measurements']}
色：{info['color']}
状態：{info['condition']}
素材：{info.get('material') or '不明'}
{style_line}

【出力形式】必ず以下の形式で出力してください。

【タイトル】
（40文字ちょうどになるまで使い切ること。文章にせず、検索されやすい単語を並べる形にする。アイテム名、サイズ、色、ブランド名（英語表記とカタカナ表記の両方が一般的なら両方）、素材、デザインの特徴（【説明文】に書く内容から検索されそうな単語を拾う）を、優先度が高い順に並べて40文字に収まるだけ詰め込む。優先度が低い単語（素材・デザインの特徴）から先に削って調整すること）

【説明文】
（商品の魅力を3つのポイントに絞り、1ポイント1行、各行の先頭に「✅」を付けて箇条書きにすること。長い文章は読まれないため、1行は20〜30文字程度の短い一文にまとめる。ポイントはデザインの特徴・素材感・着こなし方・季節感などから、その商品に合うものを3つ選ぶ。着こなし方を提案する行は【当店のコンセプト】のトーン（アイテムで着飾るより、色や小物の使い方・着る人の雰囲気で魅せる、抜け感のある提案）を意識すること。ただし色・素材・状態など商品自体の事実は正確に書き、コンセプトに寄せるために事実を誇張・変更しないこと。見出し・実寸・注意書きは書かない。行頭の✅以外の記号（✨・☺・☘など）は文中で使わず、ごちゃごちゃしないようにする）

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

    title = enforce_title_length(extract_section(text, 'タイトル'))
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

    return {"title": title, "description": description, "hashtags": hashtags}

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
