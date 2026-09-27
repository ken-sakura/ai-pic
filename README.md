# Image Generation Toolkit (完全ローカルモデル参照版)

Stable Diffusion 1.5 (AnyLoRA + SD-VAE) を用いた、**完全ローカル・オフライン完結型のバッチ画像生成ツールキット**です。

推論スクリプトは外部ネットワークへの通信を一切遮断したオフラインモードで動作し、ローカルにダウンロードされたキャッシュモデルのみを参照します。  
タスク設定用の JSON ファイルによりプロンプトや解像度、シード値を管理でき、複数ターミナルや tmux による並行バッチ生成にも対応しています。

---

## 📁 ディレクトリ構成

```text
.
├── configs/                   # 画像生成タスク設定ファイル (JSON)
│   ├── tasks.json             # 標準・キャラクター生成タスク設定例
│   └── back.json              # 1920×1080背景生成タスク設定例
├── outputs/                   # 生成画像の出力先ディレクトリ (.gitkeep)
├── generate.py                # メイン画像生成スクリプト
├── generate_background.py     # 1920×1080 背景画像生成特化スクリプト
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

本ツールキットの生成スクリプト（`generate.py`, `generate_background.py`）は、セキュリティと再現性のため **完全オフラインモード** (`HF_HUB_OFFLINE=1`, `local_files_only=True`) で動作します。  
**初回の画像生成前に、必ずローカルへモデルをダウンロードしてください。**

### 方法 A: 付属のダウンロードスクリプトを使用（推奨）

付属の `download_models.py` を実行すると、必要なモデルと VAE を Hugging Face Hub から取得し、ローカルキャッシュに保存・検証します。

```bash
python download_models.py
```

- **ダウンロード対象**:
  - ベースモデル: `Lykon/AnyLoRA`
  - VAE: `stabilityai/sd-vae-ft-mse`
- 必要なファイルのみを効率的にダウンロードするため、無駄なストレージを消費しません。

### 方法 B: `huggingface-cli` コマンドで個別にダウンロード

ターミナルから直接 `huggingface-cli` を使用してダウンロードすることも可能です。

```bash
# AnyLoRA モデルのダウンロード
huggingface-cli download Lykon/AnyLoRA

# SD-VAE のダウンロード
huggingface-cli download stabilityai/sd-vae-ft-mse
```

### 📂 モデルの保存場所と完全オフライン環境への移行

- ダウンロードされたモデルは、ユーザーディレクトリのキャッシュに格納されます。
  - **保存先パス**: `~/.cache/huggingface/hub/`
    - `models--Lykon--AnyLoRA/`
    - `models--stabilityai--sd-vae-ft-mse/`
- **オフラインマシンへ移行する場合**:
  上記フォルダをアーカイブ（zip/tar）して移行先マシンの `~/.cache/huggingface/hub/` に展開するだけで、完全にインターネット接続のない環境でも動作します。

### 🔄 別のモデルを使用したい場合

1. 新しいモデルをローカルキャッシュにダウンロード:
   ```bash
   python download_models.py --model-id <HuggingFaceリポジトリID>
   ```
2. `generate.py` や `generate_background.py` 内の `MODEL_ID` 定数を変更してください。

---

## 🎨 使い方

### 1. 標準画像生成 (`generate.py`)

キャラクターや汎用イラストを生成するスクリプトです。Euler a (Euler Ancestral) サンプラーを採用し、アニメ調イラストを高品質に生成します。

```bash
# デフォルト設定 (configs/tasks.json) で生成
python generate.py

# 任意の設定JSONファイルを指定して実行
python generate.py configs/my_task.json

# 出力先ディレクトリやデバイスの明示指定
python generate.py configs/tasks.json -o custom_outputs/ --device cuda
```

#### 主な引数オプション
- `task_file` (位置引数) または `-f / --config / --task-file`: 読み込む設定 JSON のパス (デフォルト: `configs/tasks.json`)
- `-o / --output-dir`: 生成画像の保存先ディレクトリ (デフォルト: `outputs/`)
- `--device {auto,cuda,cpu}`: 実行デバイス (デフォルト: `auto`。GPU があれば自動で CUDA/FP16、なければ CPU/FP32)

---

### 2. 1920×1080 背景画像生成特化 (`generate_background.py`)

デスクトップ壁紙や配信用背景などに適した **1920×1080 高解像度背景** を生成するスクリプトです。

```bash
# デフォルト設定 (configs/back.json) で生成
python generate_background.py

# 任意の設定JSONファイルを指定して実行
python generate_background.py configs/custom_back.json
```

#### 背景特化スクリプトの特長
- **構図破綻の防止 & 低メモリ動作**:  
  SD 1.5 の学習解像度に近い `768×432` で構図・光・質感を生成した後、高品質 **Lanczos リサンプリング** で `1920×1080` へアップスケールします。CPU 環境でもメモリ消費 3〜4GB 前後で軽快に動作します。
- **高精細サンプラー**:  
  建築・自然・直線構造をシャープに描画する `DPMSolverMultistepScheduler` (DPM++ 2M Karras) を採用。
- **メモリピーク抑制**:  
  VAE Tiling / Slicing および Attention Slicing を標準で有効化。

#### 主な引数オプション
- `task_file` (位置引数) または `-f / --config`: 設定 JSON のパス (デフォルト: `configs/back.json`)
- `-o / --output-dir`: 保存先ディレクトリ (デフォルト: `outputs/`)
- `--device {auto,cuda,cpu}`: 実行デバイス (デフォルト: `auto`)
- `--gen-width`, `--gen-height`: ベース生成解像度 (デフォルト: `768`, `432`)
- `--target-width`, `--target-height`: 最終出力解像度 (デフォルト: `1920`, `1080`)

---

### 🪟 並行バッチ生成（tmux / マルチターミナル）

複数の設定ファイル（例: `tasks_a.json`, `tasks_b.json`）を作成し、別々のターミナルで実行することで、バッチ生成を並行処理できます。

```bash
# ターミナル 1
python generate.py configs/tasks_group1.json

# ターミナル 2
python generate.py configs/tasks_group2.json
```

> [!TIP]
> 各設定ファイル内で指定する `output_file`（出力ファイル名）が重複しないように設定してください。

---

## ⚙️ タスク設定ファイル (*.json) の仕様

`configs/` 配下の JSON ファイルは、タスクオブジェクトの配列として記述します。

```json
[
  {
    "id": "01_sample_girl",
    "prompt": "masterpiece, best quality, ultra-detailed, 1girl, solo, silver hair, neon street, anime illustration",
    "negative_prompt": "(worst quality, low quality:1.4), (bad anatomy:1.2), blurry, deformed, text, watermark",
    "output_file": "01_sample_girl.png",
    "width": 512,
    "height": 768,
    "steps": 24,
    "guidance_scale": 7.0,
    "seed": 1001
  }
]
```

| キー名 | 型 | 説明 |
|---|---|---|
| `id` | string | タスクの識別名（ログ出力用） |
| `prompt` | string | ポジティブプロンプト |
| `negative_prompt` | string | ネガティブプロンプト |
| `output_file` | string | 出力画像ファイル名（`outputs/` 配下に保存） |
| `width` | int | 生成幅 (px) |
| `height` | int | 生成高さ (px) |
| `steps` | int | 推論ステップ数 (推奨: 20〜28) |
| `guidance_scale` | float | CFGスケール。プロンプトへの忠実度 (推奨: 6.0〜7.5) |
| `seed` | int | 乱数シード (`-1` を指定するとランダム) |

---

## 🔧 ハードウェア要件とデバイス動作

- **GPU (NVIDIA CUDA)**:
  - CUDA が検出された場合、自動的に `cuda` デバイスおよび `torch.float16`（半精度）で動作します。VRAM 4GB 以上の GPU で快適に動作します。
- **CPU 環境**:
  - CUDA がない場合、自動的に `cpu` デバイスおよび `torch.float32` で安全にフォールバックします。
  - Attention Slicing と VAE Tiling/Slicing により、一般的なノート PC や省メモリ環境でも OOM（メモリ不足）を起こさず完走します。
- **明示的な指定**:
  - 引数 `--device cpu` または `--device cuda` を指定することで、強制的にデバイスを固定できます。

---

## ❓ トラブルシューティング

### Q. `LocalEntryNotFoundError` または `Cannot find ... in local cache` エラーが出る
**原因**: スクリプトが完全ローカルモードで動作しているため、ローカルキャッシュにモデルが存在しません。  
**解決策**: インターネット接続を確認し、まず `python download_models.py` を実行してモデルをダウンロードしてください。

### Q. 出力画像の顔や手足が崩れる
- `negative_prompt` に `(bad anatomy:1.2), (bad hands:1.2), deformed` 等を追加してください。
- `steps` を `24` 前後、`guidance_scale` を `6.5〜7.0` に調整してください。
- ベース解像度は極端に大きくせず、縦長なら `512×768`、横長なら `768×512` や `768×432` など、SD 1.5 の基本比率に合わせることをおすすめします。

---

## 📄 ライセンス
本リポジトリのスクリプトコードは MIT ライセンスです。  
※ 生成モデル（AnyLoRA 等）および生成物の取り扱いについては、各モデルのライセンス（CreativeML Open RAIL-M 等）に準拠してください。
