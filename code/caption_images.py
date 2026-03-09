#!/usr/bin/env python3
"""
Auto-caption training images using LLaVA for FLUX LoRA training.
Generates text files with the same name as each image.
Uses local cached LLaVA model.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm

# Local model path (cached in HuggingFace)
DEFAULT_LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"


def load_llava_model(model_id=DEFAULT_LLAVA_MODEL):
    """Load LLaVA model and processor from local cache."""
    print(f"Loading LLaVA model: {model_id}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    return model, processor


def caption_image(model, processor, image_path, style_name=""):
    """Generate a caption for a single image."""

    image = Image.open(image_path).convert("RGB")

    base_prompt = "Describe this image in detail for training an AI image generation model. "
    if style_name:
        base_prompt += f"This is in the style of '{style_name}'. "
    base_prompt += "Focus on the composition, colors, subjects, lighting, mood, and artistic techniques. Be concise but descriptive."

    prompt = f"USER: <image>\n{base_prompt}\nASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )

    caption = processor.decode(output[0], skip_special_tokens=True)

    # Extract just the response (after ASSISTANT:)
    if "ASSISTANT:" in caption:
        caption = caption.split("ASSISTANT:")[-1].strip()

    return caption


def process_directory(
    input_dir: str,
    output_dir: str = None,
    style_name: str = "",
    trigger_word: str = "",
    model_id: str = DEFAULT_LLAVA_MODEL
):
    """Process all images in a directory and generate captions."""

    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to caption")

    # Load model
    model, processor = load_llava_model(model_id)

    # Process each image (skip already captioned)
    skipped = 0
    for image_path in tqdm(image_files, desc="Captioning images"):
        caption_file = output_path / f"{image_path.stem}.txt"
        if caption_file.exists():
            skipped += 1
            continue
        try:
            caption = caption_image(model, processor, image_path, style_name)

            # Prepend trigger word if specified
            if trigger_word:
                caption = f"{trigger_word}, {caption}"

            # Save caption with same name as image but .txt extension
            caption_file = output_path / f"{image_path.stem}.txt"
            caption_file.write_text(caption)

            print(f"\n{image_path.name}: {caption[:100]}...")

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

    if skipped:
        print(f"\nSkipped {skipped} already-captioned images")
    print(f"\nCaptions saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Auto-caption images using LLaVA")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(IMAGES_DIR),
        help="Directory containing training images"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for captions (default: same as input)"
    )
    parser.add_argument(
        "--style", "-s",
        type=str,
        default="",
        help="Style name to include in prompts (e.g., 'stillion art style')"
    )
    parser.add_argument(
        "--trigger", "-t",
        type=str,
        default="",
        help="Trigger word to prepend to all captions (e.g., 'STILLION')"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_LLAVA_MODEL,
        help="LLaVA model to use"
    )

    args = parser.parse_args()

    process_directory(
        input_dir=args.input,
        output_dir=args.output,
        style_name=args.style,
        trigger_word=args.trigger,
        model_id=args.model
    )


if __name__ == "__main__":
    main()
