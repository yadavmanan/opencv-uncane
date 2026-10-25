"""LoRA fine-tune of an open VLM (default Qwen2-VL-2B-Instruct) on VizWiz SFT data.

Runs inside a SageMaker training job (see launch_training.py) or on any CUDA box:

    python train_vizwiz_lora.py --data-dir data/vizwiz --output-dir out/ --max-samples 2000

Input  (SM_CHANNEL_TRAIN or --data-dir): sft_train.jsonl + images/train/*.jpg from
       scripts/prepare_vizwiz.py --images train --sft
Output (SM_MODEL_DIR or --output-dir):   merged full model + processor, ready for
       a Hugging Face TGI endpoint (deploy_endpoint.py).

The target text is the same JSON contract the cloud SageMaker backend parses
({"speech", "needs_better_view", "claims_path_clear"}), so the fine-tuned model
learns the VizWiz domain (blur, odd angles, "unanswerable") in our output format.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from PIL import Image


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--data-dir", default=os.environ.get("SM_CHANNEL_TRAIN", "data/vizwiz"))
    ap.add_argument("--output-dir", default=os.environ.get("SM_MODEL_DIR", "out"))
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-image-side", type=int, default=768)
    return ap.parse_args()


class VizWizSFT(torch.utils.data.Dataset):
    def __init__(self, root: Path, max_samples: int, max_side: int):
        rows = [json.loads(l) for l in open(root / "sft_train.jsonl", encoding="utf-8")]
        rows = [r for r in rows if (root / r["image"]).exists()]
        random.Random(0).shuffle(rows)
        self.rows = rows[:max_samples] if max_samples else rows
        self.root, self.max_side = root, max_side

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.root / r["image"]).convert("RGB")
        img.thumbnail((self.max_side, self.max_side))
        return img, r["messages"]


def make_collator(processor):
    marker = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    pad_id = processor.tokenizer.pad_token_id

    def collate(batch):
        images = [b[0] for b in batch]
        texts = [processor.apply_chat_template(b[1], tokenize=False) for b in batch]
        enc = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[labels == pad_id] = -100
        # Train only on the assistant's answer: mask everything up to the last assistant marker.
        for row in range(labels.size(0)):
            ids = enc["input_ids"][row].tolist()
            start = 0
            for j in range(len(ids) - len(marker), -1, -1):
                if ids[j:j + len(marker)] == marker:
                    start = j + len(marker)
                    break
            labels[row, :start] = -100
        enc["labels"] = labels
        return enc

    return collate


def main():
    args = parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Trainer, TrainingArguments

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # older transformers
        from transformers import Qwen2VLForConditionalGeneration as AutoVLM

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoVLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    ds = VizWizSFT(Path(args.data_dir), args.max_samples, args.max_image_side)
    print(f"training on {len(ds)} VizWiz examples")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/checkpoints", num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
            bf16=True, logging_steps=10, save_strategy="no", report_to="none",
            remove_unused_columns=False, dataloader_num_workers=2, warmup_ratio=0.03, lr_scheduler_type="cosine",
        ),
        train_dataset=ds,
        data_collator=make_collator(processor),
    )
    trainer.train()

    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir)
    print(f"saved merged model to {args.output_dir}")


if __name__ == "__main__":
    main()
