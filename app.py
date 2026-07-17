import os
import re
from flask import Flask, render_template, request, jsonify
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DISCLAIMER = """✨ご購入前に、ぜひ読んでいただきたいこと✨

一点一点、着回しやすさや季節感を基準に選んで仕入れています。
コーディネートに迷ったら、お気軽にコメントください☺

古着という特性上、あらかじめ知っておいていただきたいことをまとめました。

◆サイズ表記だけでなく、必ず実寸サイズをご確認ください
古着は同じ表記サイズでも、ブランドや年代によってサイズ感が異なります。
実寸を基準にご判断いただくと、サイズ選びの失敗が少なくなります。

◆写真は加工せず、スマホで撮影したそのままの実物です
色味や質感をできるだけ実際に近い状態でお届けしたいので、あえて加工はしていません。
光の当たり方で多少印象が変わる場合があります。

◆「目立った傷・汚れなし」の基準について
古着のため、多少の使用感（軽い毛羽立ちやごく小さな汚れなど）がある場合がありますが、
パッと見て気にならない範囲は「目立った傷・汚れなし」としています。

◎ 古着ならではの風合いも楽しんでいただける方に、ぜひ手に取っていただきたいです☘
◎ 匂いや細かなダメージなど、見落としてしまう場合があります。気になる点は気軽にコメントください
◎ 発送の際はコンパクトに畳んでお送りします。到着後、軽くしわを伸ばしていただくと綺麗に着ていただけます"""

FOLLOW_DISCOUNT = """✨最後まで読んでくださってありがとうございます✨
当店をフォローしていただいた方には、ちょっとしたお値引きをさせていただいております☺
よろしければ他のお品物も覗いてみてください☘"""


def format_measurements(measurements):
    pairs = re.findall(r'([^\d\s：:]+)[：:]?(\d+)', measurements)
    if not pairs:
        return measurements
    return '\n'.join(f'✿{label}：約{num}cm' for label, num in pairs)


def extract_section(text, section):
    match = re.search(rf'【{re.escape(section)}】\s*([\s\S]*?)(?=\n【|$)', text)
    if not match:
        return ''
    return re.sub(r'^\(.*?\)\s*', '', match.group(1).strip(), flags=re.S).strip()

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
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

def generate_description(info):
    style_line = f"スタイル・雰囲気：{info['style']}" if info.get('style') else ''
    prompt = f"""
あなたはメルカリ出品のプロです。
以下の商品情報をもとに、タイトル・説明文・ハッシュタグを作成してください。

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
（40文字ちょうどになるまで使い切ること。文章にせず、検索されやすい単語を並べる形にする。ブランド名（英語表記とカタカナ表記の両方が一般的なら両方）、アイテム名、色、素材、デザインの特徴（【説明文】に書く内容から検索されそうな単語を拾う）、サイズを、優先度が高い順に並べて40文字に収まるだけ詰め込む）

【説明文】
（120〜180文字。商品の魅力（デザインの特徴・素材感・着こなし方・季節感など）が伝わる文章のみを書くこと。見出し・実寸・注意書きは書かない。1文ごとに改行を入れて読みやすくする。箇条書きにはしない。記号は✨・☺・☘の中からのみ選び、1〜2文につき1個程度、控えめに使う。✿は実寸の表記で使用しているため文章中では使わない。使いすぎてごちゃごちゃした印象にならないようにする）

【ハッシュタグ】
（5〜8個。メルカリで検索されやすいものを選ぶ）
"""
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = response.text
    except Exception as e:
        print("Gemini APIエラー:", repr(e))
        if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
            return {"error": "只今アクセスが集中しています。1分ほど待ってからもう一度お試しください。"}
        return {"error": "生成に失敗しました。もう一度お試しください。"}

    title = extract_section(text, 'タイトル')
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

    description = '\n\n'.join([
        appeal,
        hashtags,
        '\n\n'.join(detail_parts),
        DISCLAIMER,
        FOLLOW_DISCOUNT,
    ])

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
