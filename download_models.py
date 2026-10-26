#!/usr/bin/env python3
"""
download_models.py
Stable Diffusion モデルおよび VAE を Hugging Face Hub からローカルキャッシュにダウンロード・検証するスクリプト。
オフライン実行（generate.py / generate_background.py）の前に実行してモデルを準備します。
"""

import argparse
import sys
import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline

# DreamShaper v8 の公式リポジトリID
DEFAULT_MODEL_ID = "Lykon/dreamshaper-8"
DEFAULT_VAE_ID = "stabilityai/sd-vae-ft-mse"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and verify Hugging Face models for offline Stable Diffusion generation."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face repository ID for the base model (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--vae-id",
        type=str,
        default=DEFAULT_VAE_ID,
        help=f"Hugging Face repository ID for VAE (default: {DEFAULT_VAE_ID})",
    )
    parser.add_argument(
        "--skip-vae",
        action="store_true",
        help="Skip downloading the VAE model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("==================================================")
    print(" Hugging Face Model Downloader & Cache Verifier")
    print("==================================================")
    print("Models will be stored in ~/.cache/huggingface/hub/.")
    print("Once downloaded, image generation works 100% offline.\n")

    vae = None
    if not args.skip_vae and args.vae_id:
        print(f"[1/2] Fetching & Verifying VAE: {args.vae_id}")
        try:
            vae = AutoencoderKL.from_pretrained(
                args.vae_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            print("      ✓ VAE successfully loaded and cached.")
        except Exception as e:
            print(f"[Error] Failed to download VAE ({args.vae_id}): {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\n[2/2] Fetching & Verifying Pipeline: {args.model_id}")
    try:
        _ = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
        )
        print("      ✓ Pipeline successfully loaded and cached.")
    except Exception as e:
        print(f"[Error] Failed to download pipeline ({args.model_id}): {e}", file=sys.stderr)
        sys.exit(1)

    print("\n==================================================")
    print(" All required models are ready for offline inference!")
    print(" You can now run:")
    print("   python generate.py")
    print("   python generate_background.py")
    print("==================================================")


if __name__ == "__main__":
    main()
