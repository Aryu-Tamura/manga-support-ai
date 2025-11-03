# 📚 Manga Support AI

コミカライズ制作を支援するための、LLMベースの原作解析・キャラクター分析・プロット生成デモアプリです。Streamlit 上で動作し、OpenAI API を利用してテキストをチャンク分割しながらラベル付け、要約、プロット化までを一気通貫で行います。

---

## ✨ 主な機能

- **原作理解**: 任意の範囲を選び、粒度（文字数目安）を設定して要約を生成。各チャンクには個別の summary が保存されます。
- **キャラ解析**: 登場人物一覧から選択し、関連チャンクに基づくキャラクターメモを生成。
- **プロット支援**: 指定区間の全発話を保持しつつ、漫画用のシナリオ叩き台を生成し、編集と Word 形式でのダウンロードが可能。
- **プロジェクト追加**: テキストまたは EPUB をアップロードし、LLM による分割・ラベル付け・要約（summary フィールド付与）・キャラ抽出を自動実行。生成結果を JSON 形式で保存し、再利用できます。
- **原作理解の検証1**: チャンク要約をブロックとして並べ替え、LLM で再構成した要約を得る実験的 UI を提供。

---

## 🏛️ ディレクトリ構成

```bash
manga-support-ai/
├── app.py                         # Streamlit エントリーポイント
├── manga_support_ai/              # アプリ本体の Python パッケージ
│   ├── __init__.py
│   ├── config.py                  # 定数・パス設定
│   ├── llm_services.py            # OpenAI API との高レベル連携
│   ├── llm_workflow.py            # 分割＆ラベル付けワークフロー
│   ├── models.py                  # EntryRecord / ProjectData 等のモデル
│   ├── storage.py                 # プロジェクトの読み書きと状態管理
│   ├── utils.py                   # ロギング・アップロード・docx ユーティリティ
│   └── views/                     # Streamlit UI コンポーネント
│       ├── __init__.py
│       ├── add_project.py
│       ├── character.py
│       ├── original.py
│       ├── plot.py
│       └── sidebar.py
├── data/                          # サンプルと生成プロジェクトの保存先
│   ├── project1/project.json      # サンプル（旧形式ファイルも読み込み対応）
│   ├── project1/characters.json
│   ├── ...                        # 新規生成時は `data/<key>/project.json` に保存
│   └── projects_index.json        # 追加プロジェクトのインデックス（自動生成）
├── epub_utils.py                  # EPUB → テキスト変換ユーティリティ
├── requirements.txt
└── README.md
```

---

## ⚙️ セットアップ手順

1. **リポジトリを取得**
   ```bash
   git clone https://github.com/Aryu-Tamura/manga-support-ai.git
   cd manga-support-ai
   ```

2. **仮想環境を作成 & 有効化**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # macOS / Linux
   # または
   .venv\Scripts\activate        # Windows
   ```

3. **依存ライブラリをインストール**
   ```bash
   pip install -r requirements.txt
   ```

4. **OpenAI API キーを設定**
   `.env.example` をコピーして `.env` を作成し、`OPENAI_API_KEY` を記入します。
   ```bash
   cp .env.example .env
   # .env を編集
   ```

5. **アプリを起動**
   ```bash
   streamlit run app.py
   ```
   ブラウザで `http://localhost:8501` を開くと UI が表示されます。

---

## 🚀 使い方の流れ

1. サイドバーから既存プロジェクトを選択するか、`プロジェクトを追加する` を押して元テキストをアップロード。
2. アップロード後は原作理解 / キャラ解析 / プロット支援タブで結果を確認・編集。
3. 生成されたプロジェクトは `data/` 以下に `*_labeled.json` と `character_*.json` として保存され、再起動後も一覧に表示されます。

### JSON 出力形式

新形式では以下のようなデータが生成されます。

```json
{
  "summary": "作品全体の要約",
  "full_text": "本文全文",
  "entries": [
    {
      "id": 1,
      "text": "…",
      "type": "dialogue",
      "speakers": ["…"],
      "time": "present",
      "location": "教室",
      "tone": "calm",
      "emotion": "neutral",
      "action": "",
      "entities": ["ジョバンニ"],
      "source_local_span": {"start": 0, "end": 120}
    }
  ]
}
```

このフォーマットはアプリ内の全タブで使用されます。

---

## 🧱 モジュール別の役割

| モジュール | 役割 |
| --- | --- |
| `config.py` | 定数・パス設定。外部ファイルの場所や UI 既定値を集約。 |
| `models.py` | `EntryRecord` / `ProjectData` 等のデータモデルとユーティリティ。 |
| `storage.py` | プロジェクトの読込・保存、Streamlit セッション状態の管理。 |
| `llm_workflow.py` | テキスト分割とラベル付けを行う低レベルワークフロー。 |
| `llm_services.py` | OpenAI API との高レベルなやりとり（要約・キャラ解析・プロット）。 |
| `views/*.py` | Streamlit UI のレンダリング（原作理解 / キャラ解析 / プロット / 追加）。 |
| `utils.py` | ロギング初期化・アップロード処理・docx 生成などの共通関数。 |

---

## 🧪 開発メモ

- LLM 呼び出しは OpenAI Responses API を利用しています。API 呼び出し失敗時はサンプル出力にフォールバック。
- 生成されたプロジェクトは `data/projects_index.json` に登録され、自動でサイドバーに反映されます。
- `python -m compileall app.py manga_support_ai/` で構文チェック可能です。

---

## 🤝 コントリビュート

1. Issue / Pull Request をお気軽にどうぞ。
2. コード変更時は `streamlit run app.py` で挙動を確認してください。
3. LLM API を利用するため、テスト時には有効な `OPENAI_API_KEY` が必要です。

---

## 📄 ライセンス

本プロジェクトは MIT License のもとで公開されています。
