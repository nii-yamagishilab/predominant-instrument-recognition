import json
import os
import random
import torchaudio
import numpy as np
import torch
from tqdm import tqdm

prefix = 'nsynth_data/'

random.seed(39)

meta = {}
instr_dist = {}
train_lst = []
val_lst = []
for off_split in ['train', 'valid','test']:
    with open(os.path.join(prefix, 'nsynth-{}/examples.json'.format(off_split)), 'r') as f:
        meta[f'{off_split}'] = json.load(f)
    for k, v in meta[f'{off_split}'].items():
        v['official_split'] = off_split

full_meta = {**meta['train'], **meta['valid']}
full_data_lst = list(full_meta.keys())

random.shuffle(full_data_lst)
split = int(0.96 * len(full_data_lst))
train_part = full_data_lst[:split]
val_part = full_data_lst[split:]

train_split = {t: full_meta[t] for t in train_part}
val_split = {d: full_meta[d] for d in val_part}

## for train split, we perform random chunk during training, while for val split, we unfold them into [:3s] and [1s:4s]
# chunk_1 = {}
# chunk_2 = {}
# for kk, vv in val_split.items():
#     chunk_1[kk + '_a'] = vv.copy()
#     chunk_1[kk + '_a']['start'] = 0
#     chunk_1[kk + '_a']['end'] = 3 * 16000
#
#     chunk_2[kk + '_b'] = vv.copy()
#     chunk_2[kk + '_b']['start'] = 16000
#     chunk_2[kk + '_b']['end'] = 16000 + 3 * 16000

# chunk_val_split = {**chunk_1, **chunk_2}

# trim out silent samples

print('full: {}  train: {}   valid:{}'.format(len(full_meta), len(train_split), len(val_split)))
n_classes = len(list(set([vvv['instrument'] for vvv in full_meta.values()])))
print('number of classes: {} '.format(n_classes))

with open('silent_samples_train.txt','r') as f1:
    for strs in f1.readlines():
        n_strs = strs.strip().split('||')[-1]
        del train_split[n_strs]
with open('silent_samples_val.txt','r') as f1:
    for strs in f1.readlines():
        n_strs = strs.strip().split('||')[-1]
        del val_split[n_strs]

print('After Trim')
print('full: {}  train: {}   valid:{}'.format(len(full_meta), len(train_split), len(val_split)))
n_classes = len(list(set([vvv['instrument'] for vvv in full_meta.values()])))
print('number of classes: {} '.format(n_classes))
# count = 0
# for k, v in tqdm(val_split.items()):
#     signal, SR = torchaudio.load(
#         os.path.join('nsynth_data/nsynth-{}'.format(v['official_split']), 'audio', v['note_str'] + '.wav'))
#     signal = signal.squeeze(0)
#     mean_sig = torch.mean(np.abs(signal))
#     if mean_sig < 5e-4:
#         with open('silent_samples_val.txt','a') as tt:
#             tt.write(v['official_split'] + '||' + v['note_str'] + '\n')
#         count += 1
# print('[VAL]number of quiet samples: {}'.format(count))

# with open(os.path.join('metadata','nsynth_train.json'), 'w') as g:
#     json.dump(train_split, g, indent=4)
#
# with open(os.path.join('metadata','nsynth_valid.json'), 'w') as gg:
#     json.dump(val_split, gg, indent=4)

with open(os.path.join('metadata','nsynth_test.json'), 'w') as gg:
    json.dump(meta['test'], gg, indent=4)
