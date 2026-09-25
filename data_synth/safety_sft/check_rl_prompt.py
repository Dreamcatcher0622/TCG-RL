"""Verify that a converted SFT sample uses the RL prompt and has consistent
image placeholders vs. the images list."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(HERE, 'output/thyme_safety_sft_train.jsonl')
old_path = os.path.join(HERE, 'output/thyme_safety_sft_train.strong_prompt.jsonl')
rl_prompt_path = os.path.join(HERE, 'prompt_safety_rl.txt')

with open(rl_prompt_path, 'r', encoding='utf-8') as f:
    rl_prompt = f.read().rstrip()
rl_prompt_stripped = rl_prompt.replace('<image>', '')

with open(train_path, 'r', encoding='utf-8') as f:
    r_new = json.loads(f.readline())

with open(old_path, 'r', encoding='utf-8') as f:
    r_old = json.loads(f.readline())

new_sys = r_new['messages'][0]['content']
old_sys = r_old['messages'][0]['content']

print('=== SFT sample #1 comparison ===')
print(f'  old system len (strong prompt): {len(old_sys):5d} chars')
print(f'  new system len (RL prompt)   : {len(new_sys):5d} chars')
print(f'  new == RL prompt (stripped)  : {new_sys == rl_prompt_stripped}')
print()
print('=== new system: first 300 chars ===')
print(repr(new_sys[:300]))
print()
print('=== consistency check ===')
n_ph = sum(m['content'].count('<image>') for m in r_new['messages'])
print(f'  <image> placeholders in messages : {n_ph}')
print(f'  images list length               : {len(r_new["images"])}')
print(f'  match                            : {n_ph == len(r_new["images"])}')
print()
print('=== assistant preserved (last 200 chars) ===')
print(r_new['messages'][2]['content'][-200:])
