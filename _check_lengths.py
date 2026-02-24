import json, statistics
from pathlib import Path

rows = []
for f in ['output_gemini_high/maternal_health_train.jsonl', 'output_gemini_high/maternal_health_eval.jsonl']:
    for line in Path(f).read_text().strip().splitlines():
        rows.append(json.loads(line))

lengths = []
for r in rows:
    text = r['text']
    marker = '<start_of_turn>model\n'
    end_marker = '<end_of_turn>'
    idx = text.find(marker)
    if idx >= 0:
        response = text[idx+len(marker):]
        response = response[:response.rfind(end_marker)]
        lengths.append(len(response))

lengths.sort()
print(f'Total rows: {len(lengths)}')
print(f'Min: {min(lengths)}  Max: {max(lengths)}')
print(f'Median: {statistics.median(lengths):.0f}')
print(f'Mean: {statistics.mean(lengths):.0f}')
print(f'p10: {lengths[len(lengths)//10]}  p90: {lengths[len(lengths)*9//10]}')
print(f'Under 800 chars: {sum(1 for l in lengths if l < 800)} ({sum(1 for l in lengths if l < 800)/len(lengths)*100:.1f}%)')
print(f'800-1500 chars: {sum(1 for l in lengths if 800 <= l <= 1500)} ({sum(1 for l in lengths if 800 <= l <= 1500)/len(lengths)*100:.1f}%)')
print(f'Over 1500 chars: {sum(1 for l in lengths if l > 1500)} ({sum(1 for l in lengths if l > 1500)/len(lengths)*100:.1f}%)')

# Also export CSVs
import pandas as pd
base = Path('output_gemini_high')
for src_name, dst_name in [('maternal_health_train.jsonl', 'maternal_health_train.csv'),
                            ('maternal_health_eval.jsonl', 'maternal_health_eval.csv')]:
    src = base / src_name
    dst = base / dst_name
    data = [json.loads(line) for line in src.read_text().strip().splitlines()]
    pd.DataFrame(data).to_csv(dst, index=False)
    print(f'Wrote {dst} ({len(data)} rows)')
