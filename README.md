# LoRA Trainer

FLUX.1-dev LoRA fine-tuning toolkit with LLaVA auto-captioning and a Gradio web UI. Built for NVIDIA DGX Spark (128GB unified memory, CUDA 13.0) but runs on any system with a supported GPU and sufficient VRAM.

## What It Does

1. **Auto-Caption** -- Generates training captions from your images using LLaVA 1.5-7B, with optional style descriptions and trigger words
2. **Train** -- Trains a LoRA adapter on FLUX.1-dev using flow matching, with checkpoint saving and resume support
3. **Generate** -- Creates images with your trained LoRA(s), including multi-LoRA stacking

All three steps are accessible through a Gradio web interface or directly from the command line.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/mboard8070/LoRA-trainer.git
cd LoRA-trainer

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Launch the Gradio UI
./start.sh
```

The app runs on **http://localhost:7865**.

If working over SSH, forward the port:
```bash
ssh -L 7865:localhost:7865 user@your-host
```

## Project Structure

```
LoRA-trainer/
├── code/
│   ├── app.py                # Gradio web UI (captioning, training, generation)
│   ├── caption_images.py     # LLaVA auto-captioning script
│   └── train_flux_lora.py    # FLUX.1-dev LoRA training script
├── images/                   # Training images go here
├── output/
│   ├── loras/                # Trained LoRA adapters
│   └── training_status.json  # Live training progress (read by the UI)
├── models/                   # Local model weights (optional)
├── data/                     # Auxiliary data
├── start.sh                  # Launch script for the Gradio app
├── requirements.txt
└── README.md
```

## Usage

### 1. Prepare Training Images

Place your images in the `images/` directory. Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

For best results, use 10-50 images that represent the style you want to learn.

### 2. Auto-Caption with LLaVA

**Via the UI:** Go to the "Auto-Caption" tab, optionally set a style name and trigger word, then click "Run Auto-Captioning".

**Via CLI:**
```bash
source venv/bin/activate

# Basic captioning
python code/caption_images.py

# With style context and trigger word
python code/caption_images.py --style "oil painting style" --trigger "MYSTYLE"
```

This generates a `.txt` caption file alongside each image. Already-captioned images are skipped on re-runs.

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--input, -i` | Image directory | `./images` |
| `--output, -o` | Caption output directory | Same as input |
| `--style, -s` | Style name for prompt context | None |
| `--trigger, -t` | Trigger word prepended to captions | None |
| `--model, -m` | LLaVA model ID | `llava-hf/llava-1.5-7b-hf` |

### 3. Train a LoRA

**Via the UI:** Go to the "Train LoRA" tab, set your parameters, and click "Start Training". Progress updates live every 5 seconds.

**Via CLI:**
```bash
# Basic training (1000 steps, rank 16)
python code/train_flux_lora.py \
  --output_dir output/loras/my-style \
  --max_steps 1000 \
  --lora_rank 16

# Custom configuration
python code/train_flux_lora.py \
  --image_dir ./images \
  --output_dir output/loras/my-style-v2 \
  --resolution 1024 \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --max_steps 500 \
  --lora_rank 16 \
  --save_steps 250
```

**Training parameters:**
| Flag | Description | Default |
|------|-------------|---------|
| `--image_dir, -i` | Training image directory | `./images` |
| `--output_dir, -o` | Output directory for LoRA weights | `./output/loras` |
| `--model_id, -m` | Base FLUX model (HuggingFace ID) | `black-forest-labs/FLUX.1-dev` |
| `--resolution, -r` | Training resolution | `1024` |
| `--batch_size, -b` | Training batch size | `1` |
| `--gradient_accumulation, -g` | Gradient accumulation steps | `4` |
| `--learning_rate, -lr` | Learning rate | `1e-4` |
| `--max_steps, -s` | Maximum training steps | `1000` |
| `--lora_rank` | LoRA rank (higher = more capacity) | `16` |
| `--save_steps` | Save checkpoint every N steps | `250` |
| `--seed` | Random seed | `42` |
| `--resume_from` | Checkpoint directory to resume from | None |

### 4. Resume from Checkpoint

Training saves checkpoints at regular intervals (default every 250 steps). If training is interrupted, resume from the last checkpoint:

```bash
python code/train_flux_lora.py \
  --output_dir output/loras/my-style \
  --max_steps 500 \
  --resume_from output/loras/my-style/checkpoint-250
```

The step number is parsed from the checkpoint directory name. The learning rate scheduler is automatically advanced to the correct position.

### 5. Generate Images

**Via the UI:** Go to the "Generate" tab, select one or more trained LoRAs, enter a prompt, and click "Generate".

Multiple LoRAs can be stacked with weighted combination for blending styles.

**Generation parameters:**
- **Steps:** 10-50 (default 28)
- **Guidance Scale:** 1.0-10.0 (default 3.5)
- **LoRA Weight:** 0.1-2.0 per adapter (default 1.0)

## Training Details

- **Base model:** FLUX.1-dev (12B parameter flow-matching transformer)
- **Method:** LoRA (Low-Rank Adaptation) via PEFT
- **Loss:** MSE flow matching (velocity prediction)
- **Optimizer:** AdamW 8-bit (bitsandbytes) with cosine LR schedule and 100-step warmup
- **Precision:** bf16 mixed precision
- **Gradient checkpointing:** Enabled by default for memory efficiency
- **Target modules:** `to_q`, `to_k`, `to_v`, `to_out.0`, `proj_in`, `proj_out`, `ff.net.0.proj`, `ff.net.2`
- **Trainable parameters:** ~39M out of ~12B (~0.33%)

## Output Format

Trained LoRAs are saved as PEFT adapters:
```
output/loras/my-style/
├── checkpoint-250/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── checkpoint-500/
│   └── ...
└── final_lora/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

The `final_lora/` directory contains the completed training result. Checkpoints are intermediate saves for resume or comparison.

## Monitoring Training

The training script writes live progress to `output/training_status.json`, which the Gradio UI polls every 5 seconds. You can also monitor from the terminal:

```bash
# Watch progress
watch -n 5 cat output/training_status.json | python -m json.tool

# Tail the training log
tail -f output/training.log
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- ~40GB+ VRAM recommended (FLUX.1-dev is a 12B model)
- HuggingFace account with access to [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) (gated model)

### Python Dependencies

Key packages (see `requirements.txt` for full list):
- `torch` (CUDA 13.0)
- `diffusers >= 0.32.0`
- `transformers >= 4.50.0`
- `accelerate >= 1.0.0`
- `peft >= 0.14.0`
- `bitsandbytes >= 0.45.0`
- `gradio >= 4.0.0`
- `einops`, `safetensors`, `Pillow`

## Tips

- **Image count:** 10-50 images works well for style training
- **Trigger words:** Use a unique, uncommon word (e.g., "STILLION") so the model associates it specifically with your style
- **Steps:** 500-2000 depending on dataset size. More images = more steps needed
- **Rank:** 16 is a good default. Increase to 32-64 for more complex styles
- **First run:** FLUX.1-dev (~24GB) downloads from HuggingFace on first use and is cached locally
- **Long training runs:** Training is launched as a detached process -- it survives SSH disconnects and the Gradio app being restarted

## License

This project uses FLUX.1-dev which is licensed under the [FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md). Check the license terms before using trained LoRAs commercially.
