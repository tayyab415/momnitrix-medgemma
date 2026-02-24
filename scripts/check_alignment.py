#!/usr/bin/env python3
import json
from pathlib import Path

nb_path = Path(
    '/Users/tayyabkhan/Downloads/medgemma/medsiglip_surgwound_finetune.ipynb')
plan_path = Path(
    '/Users/tayyabkhan/Downloads/medgemma/medsiglip_finetune_plan_v2.md')

nb = json.loads(nb_path.read_text())
code = '\n'.join(''.join(c['source'])
                 for c in nb['cells'] if c['cell_type'] == 'code')
plan_text = plan_path.read_text()

checks = {
    'num_labels_6': ('num_labels=NUM_LABELS' in code and 'NUM_LABELS = len(LABEL_NAMES)' in code),
    'label_heads_present': all(x in code for x in ['healing_status', 'erythema', 'edema', 'infection_risk', 'urgency', 'exudate']),
    'pos_weight_matches': 'POS_WEIGHT = torch.tensor([1.42, 2.59, 6.56, 5.15, 7.42, 5.24])' in code,
    'missing_masking': ('mask = (labels >= 0).float()' in code and 'safe_labels = labels.clamp(min=0.0)' in code),
    'selective_unfreeze': ('N_UNFREEZE = 4' in code and 'for layer in encoder_layers[-N_UNFREEZE:]' in code),
    't4_batching': ('BATCH_SIZE = 4' in code and 'GRAD_ACCUM = 16' in code),
    'epochs5_lr': ('EPOCHS = 5' in code and 'LR = 5e-5' in code),
    'fp16_true': 'FP16 = True' in code,
    'centercrop_padding': 'CenterCrop(max(image.size))' in code,
    'single_gpu_pin': 'CUDA_VISIBLE_DEVICES' in code,
    'zip_fallback': ('images.zip' in code and 'zipfile.ZipFile' in code),
    'evaluation_strategy_epoch': 'evaluation_strategy="epoch"' in code,
    'metric_best_eval_loss': 'metric_for_best_model="eval_loss"' in code,
}

print('Notebook-vs-plan critical checks:')
for key, value in checks.items():
    print(f'- {key}: {value}')

print('\nPlan has metric_for_best_model=roc_auc_macro:',
      'metric_for_best_model="roc_auc_macro"' in plan_text)
print('Notebook has metric_for_best_model=eval_loss:',
      checks['metric_best_eval_loss'])
