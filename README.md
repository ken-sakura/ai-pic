# Image Generation Toolkit (完全ローカルモデル参照版)

Stable Diffusion 1.5 (**DreamShaper 8** + SD-VAE) を用いた、**完全ローカル・オフライン完結型のバッチ画像生成ツールキット**です。

推論スクリプトは外部ネットワークへの通信を遮断したオフラインモードで動作し、ローカルにダウンロードされたキャッシュモデルのみを参照します。  
設定用の JSON ファイルは **モジュール構造化フォーマット**（共通画質・キャラ設定・背景・各ポーズの分離管理）に対応しており、キャラクターの一貫性を保ちながら多彩なポーズやバリエーションを効率的にバッチ生成できます。

---

## 📁 ディレクトリ構成

```text
.
├── configs/                   # 画像生成タスク設定ファイル (JSON)
│   ├── tasks.json             # キャラクター設定1: ピンクボブ女騎士 (5ポーズ)
│   └── tasks2.json            # キャラクター設定2: 白髪の大人の魔道士 (5ポーズ)
├── outputs/                   # 生成画像の出力先ディレクトリ (.gitkeep)
├── generate.py                # メイン画像生成スクリプト (バッチ生成・オフライン実行)
├── download_models.py         # モデル事前ダウンロード＆キャッシュ検証スクリプト
├── requirements.txt           # Python依存パッケージ一覧
├── .gitignore                 # Git管理除外設定
└── README.md                  # 本ドキュメント
```

---

## 🚀 セットアップ手順

### 1. 仮想環境の作成と有効化

```bash
# 仮想環境の作成
python3 -m venv .ve

# 仮想環境の有効化
source .ve/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

---

## 📥 モデルの事前ダウンロード（重要）

本ツールキットの生成スクリプト（`generate.py`）は、セキュリティと再現性のため **完全オフラインモード** (`HF_HUB_OFFLINE=1`, `local_files_only=True`) で動作します。  
**初回の画像生成前に、必ずローカルへモデルをダウンロードしてください。**

### 方法 A: 付属のダウンロードスクリプトを使用（推奨）

付属の `download_models.py` を実行すると、必要なモデルと VAE を Hugging Face Hub から取得し、ローカルキャッシュに保存・検証します。

```bash
python download_models.py
```

- **デフォルトのダウンロード対象**:
  - ベースモデル: `Lykon/dreamshaper-8`
  - VAE: `stabilityai/sd-vae-ft-mse`
- 必要なファイルのみを効率的にダウンロードするため、無駄なストレージを消費しません。

### 方法 B: `huggingface-cli` コマンドで個別にダウンロード

```bash
# DreamShaper 8 モデルのダウンロード
huggingface-cli download Lykon/dreamshaper-8

# SD-VAE のダウンロード
huggingface-cli download stabilityai/sd-vae-ft-mse
```

### 📂 モデルの保存場所とオフラインマシンへの移行

- ダウンロードされたモデルは、ユーザーディレクトリのキャッシュに格納されます。
  - **保存先パス**: `~/.cache/huggingface/hub/`
    - `models--Lykon--dreamshaper-8/`
    - `models--stabilityai--sd-vae-ft-mse/`
- **完全スタンドアロン環境へ移行する場合**:  
  上記フォルダをアーカイブ（zip/tar）して移行先マシンの `~/.cache/huggingface/hub/` に展開するだけで、インターネット接続のない環境でも動作します。

---

## 🎨 使い方

### 画像バッチ生成 (`generate.py`)

Euler a (Euler Ancestral) サンプラーを採用し、構図とディテールに優れたイラストを高速生成します。

```bash
# 1. デフォルト設定 (configs/tasks.json) で生成
python generate.py

# 2. 別の設定ファイルを指定して生成 (例: tasks2.json)
python generate.py configs/tasks2.json

# 3. 出力先ディレクトリや実行デバイスを明示指定
python generate.py configs/tasks.json -o custom_outputs/ --device cuda
```

#### コマンドライン引数
- `task_file` (位置引数) または `-f / --config / --task-file`: 読み込む設定 JSON のパス (デフォルト: `configs/tasks.json`)
- `-o / --output-dir`: 生成画像の保存先ディレクトリ (デフォルト: `outputs/`)
- `--device {auto,cuda,cpu}`: 実行デバイス (デフォルト: `auto`。CUDA環境があれば自動で GPU/FP16、なければ CPU/FP32 で動作)

---

## ⚙️ タスク設定ファイル (*.json) の特徴と仕様

本ツールキットの設定ファイルは、プロンプトのメンテナンス性と再現性を高めるため、**モジュール構造化フォーマット**（推奨）および従来の**フラットリスト形式**の両方に対応しています（`generate.py` が自動判別）。

### 1. モジュール構造化フォーマット（推奨）

キャラクターの容姿や背景、共通のネガティブ要素を分離して定義できるため、**「同一キャラクターの複数ポーズ展開」**が極めて直感的に作成できます。

```json
{
  "common_base": {
    "positive_anatomy": "masterpiece, best quality, (anatomically correct human anatomy:1.2), ...",
    "negative_anatomy": "((bad anatomy, deformed, mutated, malformed:1.4)), ((bad hands:1.4)), ..."
  },
  "character_settings": {
    "subject": "1girl, solo, battle mage, full body composition",
    "face_hair": "slender Japanese beauty, (long white hair, side-swept loose single braid:1.25), ...",
    "armor_body": "wearing modest ash-white sleeveless tunic, high neck collar, elegant light grey open long robe, ...",
    "negative_armor": "cleavage, deep neckline, bare midriff, see-through, sheer clothing, revealing clothes"
  },
  "background_setting": {
    "positive": "ancient gothic stone ruins, night sky, cold moonlight, sharp focus, 8k",
    "negative": "bright daylight, sunny, modern city, blurry background"
  },
  "tasks": [
    {
      "id": "pose01_levitation",
      "pose": {
        "action": "(floating mid-air, dynamic levitation pose:1.2), flowing robe",
        "expression": "(calm focused gaze, serene dignified smile:1.15)"
      },
      "custom_negative": "stiff posture, standing on ground",
      "output_file": "pose01_levitation.png",
      "width": 512,
      "height": 912,
      "steps": 28,
      "guidance_scale": 8.0,
      "seed": 8001
    }
  ]
}
```

#### 構造のメリット
- **プロンプトの重複排除**: 画質タグやキャラ設定をタスクごとにコピペする必要がありません。
- **結合の自動化**: `generate.py` が自動的に `[共通画質] + [キャラ基本情報] + [背景] + [個別ポーズ・表情]` を連結して最終プロンプトを構築します。
- **ネガティブの多重防護**: 解剖学的破綻の排除 (`negative_anatomy`)、衣装・コンプライアンス違反防止 (`negative_armor`)、構図固有の除外タグ (`custom_negative`) が安全に合成されます。

---

### 2. 従来型フラットリスト形式

シンプルな 1 枚ごとのプロンプト指定も引き続きサポートしています。

```json
[
  {
    "id": "sample_task",
    "prompt": "masterpiece, best quality, 1girl, solo, silver hair, anime illustration",
    "negative_prompt": "(worst quality, low quality:1.4), (bad anatomy:1.2), blurry",
    "output_file": "sample_task.png",
    "width": 512,
    "height": 768,
    "steps": 24,
    "guidance_scale": 7.0,
    "seed": 1001
  }
]
```

---

## 🔧 ハードウェア要件と動作仕様

- **GPU (NVIDIA CUDA)**:
  - CUDA が検出された場合、自動的に `cuda` デバイスおよび `torch.float16`（半精度）で動作します。VRAM 4GB〜8GB 程度で快適に動作します。
- **CPU 環境**:
  - CUDA がない場合、自動的に `cpu` デバイスおよび `torch.float32` で安全にフォールバックします。
  - Attention Slicing が標準有効化されているため、省メモリ環境でも OOM（メモリ不足）を起こさず安定して完走します。
- **メモリ管理**:
  - 各タスクの生成ごとにガベージコレクション (`gc.collect`) および CUDA キャッシュ解放 (`torch.cuda.empty_cache`) を自動実行します。

---

## 📄 ライセンス
本リポジトリのスクリプトコードは MIT ライセンスです。  
※ 生成モデル（DreamShaper 8 等）および生成物の取り扱いについては、各モデルのライセンス（CreativeML Open RAIL-M 等）に準拠してください。
