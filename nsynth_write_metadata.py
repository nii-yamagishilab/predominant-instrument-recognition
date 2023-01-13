import json
import os
from sklearn.preprocessing import LabelEncoder

instr_fml_labels =['vocal',
                   'string',
                   'bass',
                   'synthetic_bass',
                   'flute',
                   'electronic_guitar',
                   'acoustic_guitar',
                   'synthetic_guitar',
                   'synth_lead',
                   'mallet',
                   'reed',
                   'brass',
                   'organ',
                   'electronic_keyboard',
                   'acoustic_keyboard',
                   'synthetic_keyboard',
                   ]
prefix = 'nsynth_data/'

le = LabelEncoder()
le.fit(instr_fml_labels)

meta = {}
instr_dist ={}
for split in ['train','valid', 'test']:
    with open(os.path.join(prefix,'nsynth-{}/examples.json'.format(split)),'r') as f:
        meta[f'{split}'] = json.load(f)
    for k, v in meta[f'{split}'].items():
        i_f_s = v['instrument_family_str']
        if i_f_s == 'guitar' or i_f_s == 'keyboard':
            i_f_s = v['instrument_source_str'] + '_' + i_f_s

        if i_f_s == 'bass' and v['instrument_source_str'] == 'synthetic':
            i_f_s = 'synthetic_bass'
        meta[f'{split}'][k]['instrument_family_rearrange'] = int(le.transform([i_f_s])[0])
        meta[f'{split}'][k]['instrument_family_rearrange_str'] = i_f_s

    with open(os.path.join('metadata',f'nsynth_{split}.json'), 'w') as g:
        json.dump(meta[f'{split}'], g, indent=4)
