# 📄 Document-to-Markdown Generation using Vision Language Model (QLoRA Fine-Tuning)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Setup & Installation](#setup--installation)
- [How to Run on Kaggle](#how-to-run-on-kaggle)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Gradio App](#gradio-app)

- [Links](#links)

---

## 🧠 Overview

This project fine-tunes **Qwen2-VL-2B-Instruct**, a state-of-the-art Vision Language Model (VLM), using **QLoRA (Quantized Low-Rank Adaptation)** to convert document page images into clean, structured Markdown text.

Given an image of a document (research paper, textbook page, scanned form), the model generates:
- Headings and subheadings
- Bullet point lists
- Tables
- LaTeX equations
- Code blocks
- Properly structured Markdown

### Key Highlights
- ✅ Multimodal learning (image + text)
- ✅ Parameter-efficient fine-tuning (trains only ~1% of weights)
- ✅ 4-bit quantization to fit on free Kaggle T4 x2 GPUs
- ✅ Gradio web app for easy inference
- ✅ Zero-shot vs fine-tuned comparison

---

## 📁 Project Structure

```
├── AI_ASS05_VLM_QLoRA.ipynb     # Main Jupyter Notebook (complete implementation)
├── README.md                     # This file
├── assets/
│   ├── sample_pair.png           # Sample image-markdown pair
│   ├── dataset_stats.png         # Dataset statistics plots
│   ├── loss_curves.png           # Training & validation loss curves
│   ├── val_comparison_1.png      # Validation comparison screenshots
│   ├── val_comparison_2.png
│   ├── val_comparison_3.png
│   ├── train_test_1.png          # Training image test results
│   ├── train_test_2.png
│   ├── train_test_3.png
│   ├── unseen_test_1.png         # Unseen image test results
│   ├── unseen_test_2.png
│   ├── unseen_test_3.png
│   └── zeroshot_vs_finetuned.png # Bonus comparison
└── requirements.txt              # Python dependencies
```

---

## 🏗️ Model Architecture

```
┌─────────────────────────────────────────────────────┐
│                   INPUT                             │
│         Document Page Image (PNG/JPG)               │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              VISION ENCODER                         │
│         (Qwen2-VL Visual Transformer)               │
│         Extracts visual patch embeddings            │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│         LANGUAGE DECODER (4-bit NF4)                │
│         Qwen2-VL-2B-Instruct                        │
│    ┌─────────────────────────────────────┐          │
│    │     LoRA Adapters (rank=16)         │          │
│    │  q_proj, k_proj, v_proj, o_proj     │          │
│    │  gate_proj, up_proj, down_proj      │          │
│    └─────────────────────────────────────┘          │
│         Frozen pretrained weights                   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  OUTPUT                             │
│         Structured Markdown Text                    │
└─────────────────────────────────────────────────────┘
```

---

## 📦 Dataset

**Nougat Training Dataset Example**
- 🔗 [Kaggle Dataset by zphilip](https://www.kaggle.com/datasets/zphilip/nougat-training-dataset-example)
- Originally from Meta AI's Nougat project
- Contains pairs of **document page images** and their **ground-truth Markdown**
- Documents include scientific papers, equations, tables, and figures

| Property | Value |
|----------|-------|
| Format | Image + Markdown pairs |
| Image type | PNG (document scans) |
| Split used | 80% train / 20% validation |
| Task | Image-to-text generation |

---

## ⚙️ Setup & Installation

### Requirements

```txt
transformers==4.45.0
peft==0.13.0
bitsandbytes==0.43.3
accelerate==0.34.2
datasets==3.0.1
qwen-vl-utils
gradio
evaluate
rouge_score
Pillow
tqdm
matplotlib
torch>=2.0.0
```

### Install

```bash
pip install transformers==4.45.0 peft==0.13.0 bitsandbytes==0.43.3 \
            accelerate==0.34.2 datasets==3.0.1 qwen-vl-utils \
            gradio evaluate rouge_score Pillow tqdm matplotlib
```

> ⚠️ Requires a CUDA-capable GPU with at least 16GB VRAM. Recommended: NVIDIA T4 x2 (available free on Kaggle).

---

## 🚀 How to Run on Kaggle

### Step 1 — Open Kaggle
Go to [kaggle.com](https://www.kaggle.com) and sign in.

### Step 2 — Create New Notebook
Click **Create** → **New Notebook**

### Step 3 — Import the Notebook
Click the **⋮ menu** (top right) → **Import Notebook** → Upload `AI_ASS05_VLM_QLoRA.ipynb`

### Step 4 — Add the Dataset
In the right sidebar → **Add Input** → Search `nougat training dataset` → Add the dataset by **zphilip**

### Step 5 — Enable GPU
Right sidebar → **Session Options** → **Accelerator** → Select **GPU T4 x2**
Also make sure **Internet is ON** (needed to download model from HuggingFace)

### Step 6 — Run All
Click **Run All** or press `Shift+Enter` on each cell.

### Step 7 — Save & Share
Click **Save Version** → **Save & Run All (Commit)** to get a shareable link.

---

## 🔧 Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2-VL-2B-Instruct |
| Quantization | 4-bit NF4 (double quant) |
| Compute dtype | bfloat16 |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | 5 (early stopping patience=3) |
| Batch size | 1 |
| Gradient accumulation | 4 (effective batch = 4) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine with warmup |
| Optimizer | paged_adamw_8bit |
| Image resolution | 512 px (longest side) |
| Max sequence length | 1024 tokens |
| Gradient checkpointing | Enabled |

---

## 📊 Results

### Loss Curves
Training and validation loss across epochs — see `assets/loss_curves.png`

### ROUGE Scores

| Split | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Validation | ~0.68 | ~0.52 | ~0.63 |
| Unseen | ~0.61 | ~0.45 | ~0.57 |

### Zero-Shot vs Fine-Tuned

| | Zero-Shot | Fine-Tuned |
|--|-----------|------------|
| ROUGE-L | ~0.18 | ~0.63 |
| Follows structure | ❌ | ✅ |
| Preserves tables | ❌ | ✅ |
| Outputs equations | ❌ | ✅ |

---

## 🖥️ Gradio App

The deployed app allows users to:
- 📂 Upload any document image
- ⚡ Click **Generate Markdown**
- 📝 View raw Markdown output
- 🔍 See live rendered Markdown preview
- 🔁 Try built-in sample images

Run the last cell in the notebook to launch the Gradio app. A public shareable link is generated automatically via `share=True`.
---

## 🔗 Links

| Resource | Link |
|----------|------|
| 📝 Medium Blog Post |https://medium.com/p/ec6f48f6ad0e?postPublishedType=initial|
| 💼 LinkedIn Post | https://www.linkedin.com/posts/urwa-sajid-134729248_generativeai-machinelearning-visionlanguagemodel-share-7458231960079855616-p0xN?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD1U83gBcIneLrKsHfxwQIrw807hV3UXRjs |
| 🤗 Base Model | [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| 📦 Dataset | [Nougat Dataset on Kaggle](https://www.kaggle.com/datasets/zphilip/nougat-training-dataset-example) |

---

## 📚 References

- [Qwen2-VL Paper](https://arxiv.org/abs/2409.12191)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Nougat: Neural Optical Understanding for Academic Documents](https://arxiv.org/abs/2308.13418)
- [PEFT Library by HuggingFace](https://github.com/huggingface/peft)
- [BitsAndBytes Library](https://github.com/TimDettmers/bitsandbytes)

---

<div align="center">
  Made with ❤️ for AI4009 — Generative AI | FAST-NUCES | Spring 2026
</div>
