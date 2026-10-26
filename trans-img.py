#!/usr/bin/env python3
"""
transform_image.py
inputs/input.png を入力画像として固定し、
設定ファイル（JSON）のプロンプトとパラメータに従って画像を加工・変換するスクリプト。
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

# 完全オフライン実行設定
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from diffusers import (
    AutoencoderKL,
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)
from PIL import Image

# パス定義
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "tasks_transform.json"
FALLBACK_CONFIG_PATH = BASE_DIR / "tasks_transform.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"

# 固定の入力画像パス（inputs/ または input/ 双方に対応）
DEFAULT_INPUT_IMAGE = BASE_DIR / "inputs" / "input.png"
FALLBACK_INPUT_IMAGE = BASE_DIR / "input" / "input.png"

# モデルID設定
MODEL_ID = "Lykon/dreamshaper-8"
VAE_ID = "stabilityai/sd-vae-ft-mse"

DEFAULT_STEPS = 28
DEFAULT_GUIDANCE_SCALE = 6.5
DEFAULT_STRENGTH = 0.65


def resolve_file(path_str: str, default_dir: Path = BASE_DIR) -> Path:
    """指定されたパスを解決する"""
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    p_sub = default_dir / path_str
    if p_sub.exists():
        return p_sub
    return p


def get_input_image_path(user_arg: str = None) -> Path:
    """入力画像を解決する（引数指定があれば優先、無ければ inputs/input.png）"""
    if user_arg:
        p = resolve_file(user_arg)
        if p and p.is_file():
            return p
        print(f"[Error] Specified input image not found: {user_arg}", file=sys.stderr)
        sys.exit(1)

    if DEFAULT_INPUT_IMAGE.is_file():
        return DEFAULT_INPUT_IMAGE
    if FALLBACK_INPUT_IMAGE.is_file():
        return FALLBACK_INPUT_IMAGE

    print(f"[Error] Input image not found. Please place 'input.png' in '{DEFAULT_INPUT_IMAGE.parent}'.", file=sys.stderr)
    sys.exit(1)


def prepare_image(image_path: Path, max_dim: int = 1024) -> Image.Image:
    """
    元画像を読み込み、RGB変換およびSD1.5が扱えるよう8の倍数の解像度にリサイズします。
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        w = int(round((w * scale) / 8.0)) * 8
        h = int(round((h * scale) / 8.0)) * 8
    else:
        w = (w // 8) * 8
        h = (h // 8) * 8

    return img.resize((max(w, 64), max(h, 64)), Image.Resampling.LANCZOS)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform inputs/input.png using Stable Diffusion Img2Img and local JSON tasks."
    )
    parser.add_argument(
        "task_file",
        nargs="?",
        default=None,
        help="Path to the task JSON file (default: configs/tasks_transform.json)",
    )
    parser.add_argument(
        "-f",
        "--config",
        dest="task_file_opt",
        default=None,
        help="Path to the task JSON file (flag option)",
    )
    parser.add_argument(
        "-i",
        "--input-image",
        default=None,
        help="Optional override for input image (default: inputs/input.png)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save output images (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help=f"Override transformation strength (0.0 to 1.0, default: {DEFAULT_STRENGTH})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 入力画像の確定
    input_image_path = get_input_image_path(args.input_image)

    # タスク設定ファイルの確定
    task_file_path = resolve_file(args.task_file_opt or args.task_file)
    if not task_file_path or not task_file_path.is_file():
        if DEFAULT_CONFIG_PATH.exists():
            task_file_path = DEFAULT_CONFIG_PATH
        elif FALLBACK_CONFIG_PATH.exists():
            task_file_path = FALLBACK_CONFIG_PATH
        else:
            print(f"[Error] Task file '{task_file_path}' does not exist.", file=sys.stderr)
            sys.exit(1)

    print(f"Input Image : {input_image_path}")
    print(f"Task File   : {task_file_path}")

    with open(task_file_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Device      : {device.upper()} ({dtype})")
    print(f"Model       : {MODEL_ID}")

    # パイプライン準備
    try:
        vae = AutoencoderKL.from_pretrained(
            VAE_ID,
            torch_dtype=dtype,
            local_files_only=True,
        )
        vae.enable_tiling()
        vae.enable_slicing()

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            vae=vae,
            torch_dtype=dtype,
            safety_checker=None,
            local_files_only=True,
        )
    except Exception as e:
        print(f"\n[Error] Model loading failed: {e}", file=sys.stderr)
        print("Please ensure models are cached via download_models.py", file=sys.stderr)
        sys.exit(1)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
    )
    pipe.enable_attention_slicing()
    pipe.to(device)

    # 元画像のロード & リサイズ
    init_image = prepare_image(input_image_path)
    print(f"Base Canvas : {init_image.width}x{init_image.height}")
    print(f"\nProcessing {len(tasks)} task(s)...")

    for i, task in enumerate(tasks, 1):
        task_id = task.get("id", f"task_{i}")
        steps = task.get("steps", DEFAULT_STEPS)
        scale = task.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
        strength = args.strength if args.strength is not None else task.get("strength", DEFAULT_STRENGTH)
        seed = task.get("seed", -1)

        print(f"\n[{i}/{len(tasks)}] Transforming: {task_id}")
        print(f"       Strength: {strength} | Steps: {steps} | Scale: {scale} | Seed: {seed}")

        generator = torch.Generator(device).manual_seed(seed) if seed != -1 else None

        output_image = pipe(
            prompt=task["prompt"],
            negative_prompt=task.get("negative_prompt"),
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator,
        ).images[0]

        output_filename = task.get("output_file", f"transformed_{i}.png")
        save_path = output_dir / output_filename
        output_image.save(save_path)
        print(f"Saved -> {save_path}")

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n✓ Transformation complete!")


if __name__ == "__main__":
    main()
