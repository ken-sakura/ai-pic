#!/usr/bin/env python3
"""
generate.py
Stable Diffusion (AnyLoRA + SD-VAE) を用いたローカル完結型のバッチ画像生成スクリプト。
設定JSONファイルに基づいて複数の画像生成タスクを連続実行します。
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

# 完全にローカルキャッシュ・ローカルファイルのみを参照するようオフラインモードを設定
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, EulerAncestralDiscreteScheduler

# パス定義
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "tasks.json"
FALLBACK_CONFIG_PATH = BASE_DIR / "tasks.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_ID = "Lykon/dreamshaper-8"
VAE_ID = "stabilityai/sd-vae-ft-mse"


def resolve_task_file(user_path: str = None) -> Path:
    """指定されたタスクファイルパス、またはデフォルトパスを解決する"""
    if user_path:
        p = Path(user_path)
        if not p.is_absolute():
            # カレントディレクトリまたはBASE_DIR基準で探索
            if p.exists():
                return p
            p_base = BASE_DIR / user_path
            if p_base.exists():
                return p_base
            p_config = BASE_DIR / "configs" / user_path
            if p_config.exists():
                return p_config
        return p

    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    if FALLBACK_CONFIG_PATH.exists():
        return FALLBACK_CONFIG_PATH
    return DEFAULT_CONFIG_PATH


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion based on a JSON task configuration file."
    )
    parser.add_argument(
        "task_file",
        nargs="?",
        default=None,
        help="Path to the task JSON file (default: configs/tasks.json)",
    )
    parser.add_argument(
        "-f",
        "--config",
        "--task-file",
        dest="task_file_opt",
        default=None,
        help="Path to the task JSON file (flag option)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save output images (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to run inference on (default: auto)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    task_file_path = resolve_task_file(args.task_file_opt or args.task_file)

    if not task_file_path.is_file():
        print(f"[Error] Task configuration file '{task_file_path}' does not exist.", file=sys.stderr)
        print("Please check the path or ensure configs/tasks.json exists.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading task configuration from: {task_file_path}")
    with open(task_file_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # デバイスおよびデータ型の決定
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Execution Device : {device.upper()} (dtype: {dtype})")
    print("Model Mode       : Strictly Local Cache Only (Offline Mode)")

    try:
        print(f"Loading VAE ({VAE_ID})...")
        vae = AutoencoderKL.from_pretrained(
            VAE_ID,
            torch_dtype=dtype,
            local_files_only=True,
        )

        print(f"Loading Pipeline ({MODEL_ID})...")
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            vae=vae,
            torch_dtype=dtype,
            safety_checker=None,
            local_files_only=True,
        )
    except Exception as e:
        print(f"\n[Error] Failed to load local model: {e}", file=sys.stderr)
        print("\nThe required model or VAE is not cached locally.", file=sys.stderr)
        print("Please download models first by running:\n  python download_models.py", file=sys.stderr)
        sys.exit(1)

    # アニメ調生成に適した Euler a サンプラーを適用
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe.to(device)

    print(f"\nLoaded {len(tasks)} task(s). Starting generation...")

    for i, task in enumerate(tasks, 1):
        task_id = task.get("id", f"task_{i}")
        print(f"\n[{i}/{len(tasks)}] Generating: {task_id}")

        width = task.get("width", 512)
        height = task.get("height", 768)
        steps = task.get("steps", 22)
        scale = task.get("guidance_scale", 6.5)
        seed = task.get("seed", -1)

        print(f"       Size: {width}x{height} | Steps: {steps} | Scale: {scale} | Seed: {seed}")

        generator = torch.Generator(device).manual_seed(seed) if seed != -1 else None

        image = pipe(
            prompt=task["prompt"],
            negative_prompt=task.get("negative_prompt"),
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator,
        ).images[0]

        output_filename = task.get("output_file", f"image_{i}.png")
        save_path = output_dir / output_filename
        image.save(save_path)
        print(f"Saved -> {save_path}")

        # メモリ解放
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n✓ All tasks completed successfully.")


if __name__ == "__main__":
    main()
