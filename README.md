# 🎨 NarrAIve  
AI-Powered Comicization Support Tool for Editors & Creators  

---

## 🧩 Overview

**NarrAIve** は、小説やライトノベルなどの長文テキストをもとに、  
物語の流れを「コマ」単位で分割し、各カットに対して  
**登場人物・感情トーン・シーン情報** などのラベルを自動付与するAIツールです。  

この構造化データを利用して、  
プロット作成・ネーム作成など**漫画制作工程をAIでサポート**することを目指しています。  

---

## 🧠 What’s New in This Version

- ✅ 長文（1万文字以上）の分割・ラベル付与に対応  
- ✅ LLMでの分割＋メタ情報付与を安定化  
- ✅ JSONプロジェクトの**保存・再読み込み機能**を実装  
- ✅ `.env` による安全なAPIキー管理  
- ✅ Streamlit UIで分割結果をリアルタイム確認  
- ⚙️ 今後追加予定：
  - プロット表現の候補提案機能  
  - キャラ別要約（RAG的参照）  
  - EPUBファイル解析による自動入力  

---

## 🏗️ Project Structure

```bash
NarrAIve/
├── app.py                    # Streamlitメインアプリ
├── .env                      # OpenAIキー（Git追跡外）
├── .env.example              # APIキー設定サンプル
├── comicizer_project.json    # 検証用サンプルデータ
├── README.md                 # このファイル
├── requirements.txt          # 依存ライブラリ一覧
└── .gitignore                # 機密・環境ファイル除外
```

了解です ✅
以下は **「⚙️ Setup Instructions」以下の部分だけ**を、
そのまま `README.md` にコピペしても GitHub で正しく表示されるように
完全な Markdown 構文で整えた最新版です👇

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Aryu-Tamura/NarrAIve.git
cd NarrAIve
```

---

### 2️⃣ Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# or
.venv\Scripts\activate      # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Set Your API Key

`.env.example` をコピーして `.env` を作成します。

```bash
cp .env.example .env
```

そして `.env` を開いて、自分の OpenAI APIキーを設定します：

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

> ⚠️ `.env` ファイルは `.gitignore` に登録済みです。
> GitHubにアップロードしないでください。

---

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

実行後、コンソールに出るURLをクリックするとWeb UIが開きます：

```
Local URL: http://localhost:8501
```

---

## 🧩 How It Works

1️⃣ **テキストを入力 or JSONをアップロード**
　→ 小説やラノベ本文をそのままペースト可能。

2️⃣ **LLMが自動でカット分割＋ラベル付与**
　→ 各コマ（約200文字単位）に登場人物・感情・シーンなどを自動追加。

3️⃣ **結果をJSONとして保存可能**
　→ 再アップロードして続きから編集・再生成が可能。

---

## 📦 Example Data

本リポジトリには、サンプルとして
`comicizer_project.json`（ラベル付きテキストデータ）が同梱されています。
Streamlit UIの「📂 JSONを読み込む」からアップロードすることで、
即座にデモを再現できます。

---

## 🧠 Development Notes

* `.env` は GitHub にアップロードしないでください。
* Streamlit上で分割結果を確認後、JSONで保存・再利用が可能です。
* 長文テキストを扱う場合、処理には数分かかることがあります。

---

