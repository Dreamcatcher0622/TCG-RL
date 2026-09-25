"""Print the structure of the first trajectory in a distillation output file.

Usage: python peek_traj.py [path/to/trajectories.jsonl]
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, 'output', 'full_trajectories.jsonl')

with open(PATH) as f:
    r = json.loads(f.readline())
print('question_id:', r['question_id'])
print('subset:', r['subset'])
print('gt_label:', r['gt_label'])
print('predicted:', r['predicted'])
print('n_iterations:', r['n_iterations'])
print('n_code_blocks:', r['n_code_blocks'])
print('n_sandbox_images:', r['n_sandbox_images'])
print('---conversation structure---')
for i, msg in enumerate(r['conversation']):
    print(f'[{i}] role={msg["role"]}, {len(msg["content"])} content items:')
    for j, item in enumerate(msg['content']):
        if item['type'] == 'text':
            snippet = item['text'][:200].replace('\n', ' ')
            print(f'    - text: {snippet!r}...')
        elif item['type'] == 'image':
            print(f'    - image: {item["image"]}')
