#!/usr/bin/env python3
"""Final validation of the patched notebook."""
import json

NB = "/Users/tayyabkhan/Downloads/medgemma/medsiglip_surgwound_finetune.ipynb"

with open(NB) as f:
    nb = json.load(f)

all_code = ""
for c in nb["cells"]:
    if c["cell_type"] == "code":
        all_code += "".join(c["source"]) + "\n"

checks = {
    "No torchvision import": "from torchvision" not in all_code,
    "No total_mem (uses total_memory)": ".total_mem " not in all_code,
    "No _base_transform": "_base_transform" not in all_code,
    "No _train_transform": "_train_transform" not in all_code,
    "No _apply_zero_pad_and_transform": "_apply_zero_pad_and_transform" not in all_code,
    "Uses PILImage.open": "PILImage.open" in all_code,
    "Defines _process_image": "def _process_image" in all_code,
    "Uses PILImage": "from PIL import Image as PILImage" in all_code,
    "Uses explicit Features": "Features({" in all_code,
    "Defines BASE_PATH": "BASE_PATH" in all_code,
    "Has LABELS_CSV": "LABELS_CSV" in all_code,
    "Has IMAGES_DIR": "IMAGES_DIR" in all_code,
    "Has image_processor": "image_processor" in all_code,
    "Has masked_bce_loss": "masked_bce_loss" in all_code,
    "Has WoundClassificationTrainer": "WoundClassificationTrainer" in all_code,
    "Has total_memory": "total_memory" in all_code,
    "Has evaluation_strategy": "evaluation_strategy" in all_code,
}

all_ok = True
for name, ok in checks.items():
    print(f"  {'✓' if ok else '✗'} {name}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print("=" * 50)
    print("ALL CHECKS PASSED ✓")
    print("=" * 50)
else:
    print("=" * 50)
    print("SOME CHECKS FAILED ✗")
    print("=" * 50)
