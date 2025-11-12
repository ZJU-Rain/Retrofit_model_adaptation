"""
Base-Model + Adapter -> Full Weights
"""

import os
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# ========== Path Arguments ==========
base_model_path   = "/Base-Model"      # Base-Model weights folder (contains config.json)
adapter_path      = "/Adapter"  # adapter folder (contains adapter_config.json)
save_dir          = "/Save-Dir"    # Directory to save the merged full model
device            = "cuda"
# ====================================

os.makedirs(save_dir, exist_ok=True)

print("1) Loading base model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
            device_map={"": device}
        )

print("2) Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

print("3) Attaching adapter...")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

print("4) Merging weights...")
model = model.merge_and_unload()

print("5) Saving pytorch_model.bin...")
model.save_pretrained(save_dir, safe_serialization=False)
tokenizer.save_pretrained(save_dir)

print(f"6) Done, pytorch_model.bin saved to → {save_dir}")