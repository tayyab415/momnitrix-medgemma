#!/usr/bin/env python3
"""Validate the fixed notebook for stale references and structure."""

import json
import sys

NB_PATH = "medsiglip_surgwound_finetune.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Total cells: {len(cells)}")
print(f"  Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code:     {sum(1 for c in cells if c['cell_type'] == 'code')}")
print()

# Check for BAD references that should have been removed
bad_refs = [
    "from torchvision",
    ".total_mem ",
    "_base_transform",
    "_train_transform",
    "InterpolationMode",
    "CenterCrop",
    "from PIL import Image\n",  # old-style; should be PILImage now
]

issues = []
for i, cell in enumerate(cells):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    for ref in bad_refs:
        if ref in src:
            issues.append(f"  Cell {i+1}: contains '{ref.strip()}'")

if issues:
    print("BAD REFERENCES FOUND:")
    for issue in issues:
        print(issue)
else:
    print("✓ No stale torchvision/old-API references found")

# Check all code cells reference the correct variable names
print()
all_code = ""
for cell in cells:
    if cell["cell_type"] == "code":
        all_code += "".join(cell["source"]) + "\n"

good_refs = {
    "_process_image": False,
    "BASE_PATH": False,
    "LABELS_CSV": False,
    "IMAGES_DIR": False,
    "PILImage": False,
    "Features(": False,
    "image_processor": False,
    "masked_bce_loss": False,
    "WoundClassificationTrainer": False,
}

for ref in good_refs:
    if ref in all_code:
        good_refs[ref] = True

print("REQUIRED REFERENCES:")
all_ok = True
for ref, found in good_refs.items():
    status = "✓" if found else "✗"
    print(f"  {status} {ref}")
    if not found:
        all_ok = False

# Check inference demo uses _process_image not _apply_zero_pad_and_transform
print()
if "_apply_zero_pad_and_transform" in all_code:
    print("✗ Still references _apply_zero_pad_and_transform (removed)")
    all_ok = False
else:
    print("✓ No references to removed _apply_zero_pad_and_transform")

if "Image.open" in all_code and "PILImage.open" not in all_code:
    print("✗ Uses Image.open instead of PILImage.open")
    all_ok = False
else:
    print("✓ Uses PILImage.open correctly")

print()
if all_ok and not issues:
    print("═" * 40)
    print("ALL CHECKS PASSED ✓")
    print("═" * 40)
else:
    print("═" * 40)
    print("SOME CHECKS FAILED ✗")
    print("═" * 40)
    sys.exit(1)
