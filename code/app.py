#!/usr/bin/env python3
"""
Stillion AI - FLUX LoRA Training & Inference App

A Gradio-based interface for:
1. Auto-captioning training images with LLaVA
2. Training FLUX.1-dev LoRA for art styles
3. Generating images with trained LoRA

Images should be placed in the 'images' directory before running.
"""

import os
import re
import json
import signal
import subprocess
import time
from pathlib import Path
import gradio as gr
import torch
from PIL import Image

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
OUTPUT_DIR = PROJECT_ROOT / "output"
LORA_DIR = OUTPUT_DIR / "loras"
TRAINING_STATUS_FILE = OUTPUT_DIR / "training_status.json"

# Ensure directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR.mkdir(parents=True, exist_ok=True)


def read_training_status():
    """Read the current training status from the status file."""
    if not TRAINING_STATUS_FILE.exists():
        return None
    try:
        status = json.loads(TRAINING_STATUS_FILE.read_text())
        # Check if the training process is still alive
        pid = status.get("pid")
        if pid:
            try:
                os.kill(pid, 0)  # signal 0 = check if process exists
                status["alive"] = True
            except OSError:
                status["alive"] = False
        else:
            status["alive"] = False
        return status
    except (json.JSONDecodeError, IOError):
        return None


def format_training_status(status):
    """Format training status as a markdown string for display."""
    if status is None:
        return "No training in progress."

    phase = status.get("phase", "unknown")
    step = status.get("step", 0)
    total = status.get("total_steps", 0)
    loss = status.get("loss")
    lr = status.get("lr")
    elapsed = status.get("elapsed", "")
    eta = status.get("eta", "")
    speed = status.get("speed", "")
    lora_name = status.get("lora_name", "")
    alive = status.get("alive", False)
    error = status.get("error")

    if phase == "complete":
        pct = 100
        bar = "=" * 30
        state = "COMPLETE"
    elif phase in ("initializing", "loading_model"):
        pct = 0
        bar = "-" * 30
        state = phase.replace("_", " ").upper()
    elif total > 0:
        pct = int(step / total * 100)
        filled = int(pct / 100 * 30)
        bar = "=" * filled + ">" + "-" * (29 - filled)
        state = "TRAINING" if alive else "INTERRUPTED"
    else:
        pct = 0
        bar = "-" * 30
        state = phase.upper()

    lines = [f"### {state}: {lora_name}"]
    lines.append(f"```")
    lines.append(f"[{bar}] {pct}%  Step {step}/{total}")
    lines.append(f"```")

    details = []
    if loss is not None:
        details.append(f"**Loss:** {loss:.4f}" if isinstance(loss, float) else f"**Loss:** {loss}")
    if lr:
        details.append(f"**LR:** {lr}")
    if speed:
        details.append(f"**Speed:** {speed}")
    if elapsed:
        details.append(f"**Elapsed:** {elapsed}")
    if eta and phase not in ("complete",):
        details.append(f"**ETA:** {eta}")
    if details:
        lines.append(" | ".join(details))

    if error:
        lines.append(f"\n**Error:** {error}")
    elif not alive and phase not in ("complete",):
        lines.append(f"\nTraining process is no longer running (interrupted or crashed).")

    return "\n".join(lines)


def get_training_status_display():
    """Get formatted training status for the UI."""
    status = read_training_status()
    if status is None:
        return "No recent training activity."
    return format_training_status(status)


def is_training_running():
    """Check if a training process is currently active."""
    status = read_training_status()
    if status is None:
        return False
    return status.get("alive", False) and status.get("phase") not in ("complete",)


def get_image_count():
    """Count images in training directory."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    count = sum(1 for f in IMAGES_DIR.iterdir() if f.suffix.lower() in extensions)
    return count


def get_caption_count():
    """Count caption files in training directory."""
    count = sum(1 for f in IMAGES_DIR.glob("*.txt"))
    return count


def get_available_loras():
    """Get list of trained LoRA models."""
    loras = []
    if LORA_DIR.exists():
        for path in LORA_DIR.iterdir():
            if path.is_dir() and (path / "adapter_config.json").exists():
                loras.append(path.name)
            elif path.is_dir():
                # Check for final_lora subdirectory
                final_lora = path / "final_lora"
                if final_lora.exists() and (final_lora / "adapter_config.json").exists():
                    loras.append(path.name)
    return loras if loras else ["No LoRAs trained yet"]


def get_status():
    """Get current status of images and captions."""
    img_count = get_image_count()
    cap_count = get_caption_count()
    loras = get_available_loras()
    lora_count = len(loras) if loras[0] != "No LoRAs trained yet" else 0

    return f"""## Current Status

**Images:** {img_count} in `{IMAGES_DIR}`
**Captions:** {cap_count} (.txt files)
**Trained LoRAs:** {lora_count}

### Image Directory
```
{IMAGES_DIR}
```

Drop your training images here before running captioning.
"""


def run_captioning(style_name, trigger_word, progress=gr.Progress()):
    """Run LLaVA captioning on training images."""
    image_count = get_image_count()
    if image_count == 0:
        return f"No images found in {IMAGES_DIR}\n\nPlease add images to the directory first."

    progress(0, desc="Starting captioning...")

    cmd = [
        str(PROJECT_ROOT / "venv" / "bin" / "python"),
        str(PROJECT_ROOT / "code" / "caption_images.py"),
        "--input", str(IMAGES_DIR),
    ]

    if style_name:
        cmd.extend(["--style", style_name])
    if trigger_word:
        cmd.extend(["--trigger", trigger_word])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=7200  # 2 hour timeout
        )

        if result.returncode == 0:
            caption_count = get_caption_count()
            return f"Captioning complete!\n\nGenerated {caption_count} captions for {image_count} images.\n\n{result.stdout[-2000:]}"
        else:
            return f"Error during captioning:\n{result.stderr[-2000:]}"

    except subprocess.TimeoutExpired:
        return "Captioning timed out after 30 minutes."
    except Exception as e:
        return f"Error: {str(e)}"


def run_training(
    lora_name,
    resolution,
    batch_size,
    learning_rate,
    max_steps,
    lora_rank,
):
    """Launch FLUX LoRA training as a background process."""
    if is_training_running():
        return "Training is already in progress. Check the status panel below."

    image_count = get_image_count()
    caption_count = get_caption_count()

    if image_count == 0:
        return f"No images found in {IMAGES_DIR}\n\nPlease add training images first."

    if caption_count == 0:
        return "No captions found. Please run auto-captioning first."

    if not lora_name or not lora_name.strip():
        return "Please enter a unique LoRA name."

    lora_name = lora_name.strip()
    output_path = LORA_DIR / lora_name

    if output_path.exists():
        return (
            f"A LoRA named '{lora_name}' already exists.\n\n"
            f"Choose a different name to avoid overwriting it "
            f"(e.g., {lora_name}-v2)."
        )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        str(PROJECT_ROOT / "venv" / "bin" / "python"), "-u",
        str(PROJECT_ROOT / "code" / "train_flux_lora.py"),
        "--image_dir", str(IMAGES_DIR),
        "--output_dir", str(output_path),
        "--resolution", str(int(resolution)),
        "--batch_size", str(int(batch_size)),
        "--learning_rate", str(learning_rate),
        "--max_steps", str(int(max_steps)),
        "--lora_rank", str(int(lora_rank)),
    ]

    # Launch as detached background process — survives page refresh
    log_file = OUTPUT_DIR / "training.log"
    with open(log_file, "w") as log_f:
        process = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
            start_new_session=True,
        )

    return (
        f"Training launched (PID {process.pid})\n\n"
        f"LoRA: {lora_name}\n"
        f"Steps: {int(max_steps)}, Rank: {int(lora_rank)}, LR: {learning_rate}\n"
        f"Resolution: {int(resolution)}, Batch: {int(batch_size)}\n\n"
        f"Monitor progress in the status panel below — it updates on refresh.\n"
        f"Log: {log_file}"
    )


def resolve_lora_path(lora_name):
    """Resolve the path to a LoRA's weights, preferring final_lora/."""
    lora_path = LORA_DIR / lora_name / "final_lora"
    if not lora_path.exists():
        lora_path = LORA_DIR / lora_name
    return lora_path


def generate_image(prompt, lora_names, lora_weight, num_steps, guidance_scale, seed, progress=gr.Progress()):
    """Generate an image using one or more stacked LoRAs."""
    if not lora_names:
        return None, "Please select at least one LoRA."

    progress(0, desc="Loading model...")

    try:
        from diffusers import FluxPipeline, FluxTransformer2DModel
        from peft import PeftModel

        # Load base transformer
        progress(0.1, desc="Loading transformer...")
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

        # Load LoRA(s)
        progress(0.2, desc=f"Loading {len(lora_names)} LoRA(s)...")
        first_path = resolve_lora_path(lora_names[0])
        transformer = PeftModel.from_pretrained(transformer, str(first_path), adapter_name=lora_names[0])

        for name in lora_names[1:]:
            lora_path = resolve_lora_path(name)
            transformer.load_adapter(str(lora_path), adapter_name=name)

        # Stack multiple LoRAs with weighted combination
        if len(lora_names) > 1:
            weights = [float(lora_weight)] * len(lora_names)
            transformer.add_weighted_adapter(
                adapters=lora_names,
                weights=weights,
                combination_type="linear",
                adapter_name="stacked",
            )
            transformer.set_adapter("stacked")

        transformer = transformer.merge_and_unload()

        # Load pipeline with LoRA-merged transformer
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")

        progress(0.3, desc="Generating image...")

        # Set seed
        generator = torch.Generator("cuda").manual_seed(int(seed))

        # Generate
        image = pipe(
            prompt=prompt,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        ).images[0]

        progress(1.0, desc="Done!")

        lora_info = ", ".join(lora_names)
        return image, f"Generated with seed {seed} | LoRAs: {lora_info} (weight: {lora_weight})"

    except Exception as e:
        return None, f"Error: {str(e)}"


def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(title="Stillion AI - FLUX LoRA Training") as app:
        gr.Markdown("# Stillion AI - FLUX LoRA Training")
        gr.Markdown("Train custom art styles on FLUX.1-dev with LLaVA auto-captioning")

        with gr.Tabs():
            # Tab 1: Status
            with gr.TabItem("Status"):
                status_md = gr.Markdown(get_status())
                refresh_btn = gr.Button("Refresh Status")
                refresh_btn.click(fn=get_status, outputs=[status_md])

            # Tab 2: Auto-Caption
            with gr.TabItem("1. Auto-Caption"):
                gr.Markdown("### Generate captions using LLaVA")
                gr.Markdown(f"Images in: `{IMAGES_DIR}`")

                with gr.Row():
                    style_name = gr.Textbox(
                        label="Style Name (optional)",
                        placeholder="e.g., stillion art style",
                        value=""
                    )
                    trigger_word = gr.Textbox(
                        label="Trigger Word (optional)",
                        placeholder="e.g., STILLION",
                        value=""
                    )

                caption_btn = gr.Button("Run Auto-Captioning", variant="primary")
                caption_output = gr.Textbox(
                    label="Captioning Output",
                    lines=15,
                    interactive=False
                )

                caption_btn.click(
                    fn=run_captioning,
                    inputs=[style_name, trigger_word],
                    outputs=[caption_output]
                )

            # Tab 3: Train LoRA
            with gr.TabItem("2. Train LoRA"):
                gr.Markdown("### Train FLUX.1-dev LoRA")

                # Live training status panel
                training_status_md = gr.Markdown(
                    value=get_training_status_display,
                    every=5,  # auto-refresh every 5 seconds
                )

                gr.Markdown("---")

                with gr.Row():
                    lora_name = gr.Textbox(
                        label="LoRA Name",
                        placeholder="e.g., stillion-v3",
                        value=""
                    )
                    resolution = gr.Slider(
                        label="Resolution",
                        minimum=512,
                        maximum=1024,
                        step=128,
                        value=1024
                    )

                with gr.Row():
                    batch_size = gr.Slider(
                        label="Batch Size",
                        minimum=1,
                        maximum=4,
                        step=1,
                        value=1
                    )
                    learning_rate = gr.Number(
                        label="Learning Rate",
                        value=1e-4
                    )

                with gr.Row():
                    max_steps = gr.Slider(
                        label="Max Training Steps",
                        minimum=100,
                        maximum=5000,
                        step=100,
                        value=1000
                    )
                    lora_rank = gr.Slider(
                        label="LoRA Rank",
                        minimum=4,
                        maximum=64,
                        step=4,
                        value=16
                    )

                train_btn = gr.Button("Start Training", variant="primary")
                train_output = gr.Textbox(
                    label="Training Output",
                    lines=8,
                    interactive=False
                )

                train_btn.click(
                    fn=run_training,
                    inputs=[lora_name, resolution, batch_size, learning_rate, max_steps, lora_rank],
                    outputs=[train_output]
                )

            # Tab 4: Generate
            with gr.TabItem("3. Generate"):
                gr.Markdown("### Generate images with your trained LoRA(s)")
                gr.Markdown("Select one or more LoRAs to stack together.")

                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A beautiful landscape in STILLION style",
                            lines=3
                        )
                        lora_select = gr.CheckboxGroup(
                            label="Select LoRA(s)",
                            choices=get_available_loras(),
                        )
                        lora_weight = gr.Slider(
                            label="LoRA Weight (per adapter)",
                            minimum=0.1,
                            maximum=2.0,
                            step=0.1,
                            value=1.0,
                        )

                        with gr.Row():
                            num_steps = gr.Slider(
                                label="Steps",
                                minimum=10,
                                maximum=50,
                                step=1,
                                value=28
                            )
                            guidance = gr.Slider(
                                label="Guidance Scale",
                                minimum=1.0,
                                maximum=10.0,
                                step=0.5,
                                value=3.5
                            )

                        seed = gr.Number(label="Seed", value=42)
                        generate_btn = gr.Button("Generate", variant="primary")
                        refresh_lora_btn = gr.Button("Refresh LoRA List")

                    with gr.Column():
                        output_image = gr.Image(label="Generated Image", type="pil")
                        gen_status = gr.Textbox(label="Status", interactive=False)

                generate_btn.click(
                    fn=generate_image,
                    inputs=[prompt, lora_select, lora_weight, num_steps, guidance, seed],
                    outputs=[output_image, gen_status]
                )

                refresh_lora_btn.click(
                    fn=lambda: gr.CheckboxGroup(choices=get_available_loras()),
                    outputs=[lora_select]
                )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7865, share=False)
