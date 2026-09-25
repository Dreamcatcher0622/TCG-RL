"""Sanity-check one row of the converted ms-swift SFT jsonl.

Usage: python debug_convert.py [path/to/thyme_safety_sft_train.jsonl]
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, 'output', 'thyme_safety_sft_train.jsonl')

with open(PATH) as f:
    r = json.loads(f.readline())

print('keys:', list(r.keys()))
print('n_messages:', len(r['messages']))
for m in r['messages']:
    if isinstance(m['content'], str):
        n_ph = m['content'].count('<image>')
        print(f"  {m['role']}: content len={len(m['content'])}, {n_ph} <image> placeholders, snippet={m['content'][:150]!r}")
print('n_images:', len(r['images']))
for p in r['images']:
    print(f"  image: {p}")
print()
print('---sanity---')
total_ph = sum(m['content'].count('<image>') for m in r['messages'] if isinstance(m['content'], str))
print(f'total placeholders in messages = {total_ph}')
print(f'total images in images[]       = {len(r["images"])}')
print(f'match: {total_ph == len(r["images"])}')
print()
print('---assistant snippet (last 400 chars)---')
print(r['messages'][2]['content'][-400:])
