#!/usr/bin/env python3
"""
generate_background.py
1920×1080 背景画像生成特化スクリプト。
SD 1.5 で構図破綻を防ぎ、かつCPU/GPUメモリに負荷をかけないベースサイズでAI生成後、
高品質 Lanczos 補間により 1920×1080 へリサイズします。
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
from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from PIL import Image

# パス定義
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "back.json"
FALLBACK_CONFIG_PATH = BASE_DIR / "back.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_ID = "Lykon/AnyLoRA"
VAE_ID = "stabilityai/sd-vae-ft-mse"

# 解像度設定
DEFAULT_TARGET_WIDTH = 1920
DEFAULT_TARGET_HEIGHT = 1080

# SD 1.5 で構図破綻を防ぎ、メモリ消費を抑えて高速生成するためのベースサイズ (16:9, 8の倍数)
DEFAULT_GEN_WIDTH = 768
DEFAULT_GEN_HEIGHT = 432

DEFAULT_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 6.5


def resolve_task_file(user_path: str = None) -> Path:
    """指定されたタスクファイルパス、またはデフォルトパスを解決する"""
    if user_path:
        p = Path(user_path)
        if not p.is_absolute():
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


def calculate_gen_size(target_w: int, target_h: int, max_dim: int = DEFAULT_GEN_WIDTH):
    """
    メモリ節約およびSD1.5の構図崩壊防止のため、
    目標解像度のアスペクト比を維持しつつ長辺 max_dim 以下の8の倍数となる生成解像度を算出します。
    """
    if max(target_w, target_h) <= max_dim:
        gw = (target_w // 8) * 8
        gh = (target_h // 8) * 8
        return max(gw, 64), max(gh, 64)

    scale = max_dim / max(target_w, target_h)
    gw = int(round((target_w * scale) / 8.0)) * 8
    gh = int(round((target_h * scale) / 8.0)) * 8
    return max(gw, 64), max(gh, 64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 1920x1080 background images using local models (Base generation + Lanczos upscale)."
    )
    parser.add_argument(
        "task_file",
        nargs="?",
        default=None,
        help="Path to the task JSON file (default: configs/back.json)",
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
    parser.add_argument(
        "--gen-width",
        type=int,
        default=None,
        help=f"Base generation width before resize (default: auto or {DEFAULT_GEN_WIDTH})",
    )
    parser.add_argument(
        "--gen-height",
        type=int,
        default=None,
        help=f"Base generation height before resize (default: auto or {DEFAULT_GEN_HEIGHT})",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=None,
        help=f"Final target width (default: {DEFAULT_TARGET_WIDTH})",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=None,
        help=f"Final target height (default: {DEFAULT_TARGET_HEIGHT})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    task_file_path = resolve_task_file(args.task_file_opt or args.task_file)

    if not task_file_path.is_file():
        print(f"[Error] Task configuration file '{task_file_path}' does not exist.", file=sys.stderr)
        print("Please check the path or ensure configs/back.json exists.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading task configuration from: {task_file_path}")
    with open(task_file_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 実行デバイスとデータ型の決定
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
        # 高解像度デコード時のメモリピークを抑制
        vae.enable_tiling()
        vae.enable_slicing()

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

    # 背景・風景の直線や質感、幾何学構造をシャープに保つ DPM++ 2M Karras サンプラーを適用
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
    )
    pipe.enable_attention_slicing()
    pipe.to(device)

    print(f"\nLoaded {len(tasks)} task(s). Starting generation...")

    for i, task in enumerate(tasks, 1):
        task_id = task.get("id", f"task_{i}")

        # 最終目標解像度の決定 (デフォルト 1920x1080)
        target_width = args.target_width or task.get("target_width") or task.get("width", DEFAULT_TARGET_WIDTH)
        target_height = args.target_height or task.get("target_height") or task.get("height", DEFAULT_TARGET_HEIGHT)

        # 構図破綻・メモリ消費を防ぐためのベース解像度の決定 (デフォルト 768x432)
        if args.gen_width and args.gen_height:
            gen_width = args.gen_width
            gen_height = args.gen_height
        elif "gen_width" in task and "gen_height" in task:
            gen_width = task["gen_width"]
            gen_height = task["gen_height"]
        else:
            gen_width, gen_height = calculate_gen_size(target_width, target_height, max_dim=DEFAULT_GEN_WIDTH)

        steps = task.get("steps", DEFAULT_STEPS)
        scale = task.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
        seed = task.get("seed", -1)

        print(f"\n[{i}/{len(tasks)}] Generating: {task_id}")
        print(f"       Base Gen: {gen_width}x{gen_height} -> Target: {target_width}x{target_height} (Lanczos upscale)")
        print(f"       Scheduler: DPM++ 2M Karras | Steps: {steps} | Scale: {scale} | Seed: {seed}")

        generator = torch.Generator(device).manual_seed(seed) if seed != -1 else None

        # --- Base Image Generation (SD 1.5) ---
        print(f"       Generating base image ({gen_width}x{gen_height})...")
        base_image = pipe(
            prompt=task["prompt"],
            negative_prompt=task.get("negative_prompt"),
            width=gen_width,
            height=gen_height,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator,
        ).images[0]

        # --- High Quality Resize (Lanczos) to Target Resolution ---
        if base_image.size != (target_width, target_height):
            print(f"       Upscaling to {target_width}x{target_height} via Lanczos resampling...")
            final_image = base_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        else:
            final_image = base_image

        output_filename = task.get("output_file", f"back_{i}.png")
        save_path = output_dir / output_filename
        final_image.save(save_path)
        print(f"Saved -> {save_path} ({final_image.size[0]}x{final_image.size[1]})")

        # メモリ解放
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n✓ All tasks finished successfully.")


if __name__ == "__main__":
    main()
