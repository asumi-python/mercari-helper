import os
from flask import Flask, render_template, request, jsonify
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_description(info):
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
購入時期：{info.get('purchased') or '不明'}
使用回数：{info.get('usage') or '不明'}
購入場所：{info.get('place') or '不明'}

【出力形式】必ず以下の形式で出力してください。

【タイトル】
（40文字以内。ブランド名・アイテム・色・サイズを含める）

【説明文】
（150〜200文字。箇条書きは使わず自然な文章で。状態の良さと信頼感が伝わるように。最後に「ご不明な点はお気軽にコメントください」で締める）

【ハッシュタグ】
（5〜8個。メルカリで検索されやすいものを選ぶ）
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    info = request.json
    result = generate_description(info)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
