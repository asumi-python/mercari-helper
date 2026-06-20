import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_input(label, required=True):
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        if not required:
            return ""
        print("　※必須項目です。入力してください。")

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
素材：{info['material'] or '不明'}
購入時期：{info['purchased'] or '不明'}
使用回数：{info['usage'] or '不明'}
購入場所：{info['place'] or '不明'}

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

def main():
    print("=" * 40)
    print("　　メルカリ出品サポートツール")
    print("=" * 40)
    print()

    info = {
        "brand":        get_input("ブランド名（必須）"),
        "item":         get_input("アイテム（例：Tシャツ、パンツ）（必須）"),
        "size":         get_input("サイズ表記（例：M、L、38）（必須）"),
        "measurements": get_input("実寸（例：肩幅43 身幅50 着丈65）（必須）"),
        "color":        get_input("色（必須）"),
        "condition":    get_input("状態（例：未使用、良い、やや傷あり）（必須）"),
        "material":     get_input("素材（任意、わからなければそのままEnter）", required=False),
        "purchased":    get_input("購入時期（任意、例：2年前）", required=False),
        "usage":        get_input("使用回数（任意、例：3回）", required=False),
        "place":        get_input("購入場所（任意、例：公式サイト）", required=False),
    }

    print()
    print("説明文を生成中...")
    print()

    description = generate_description(info)

    print("=" * 40)
    print("【生成された説明文】")
    print("=" * 40)
    print(description)
    print("=" * 40)

if __name__ == "__main__":
    main()
