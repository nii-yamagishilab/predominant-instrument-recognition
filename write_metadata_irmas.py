import json
import glob
import os
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm


# 0: hard onset; 1: soft onset; 2: voice;
onset_to_chr = {0: 'hard_onset', 1: 'soft_onset', 2: 'voice'}
onset_group = {
   'gel': 0,
   'gac': 0,
   'pia': 0,
   'org': 0,
   'cel': 1,
   'vio': 1,
   'cla': 1,
   'sax': 1,
   'tru': 1,
   'flu': 1,
   'voi': 2,
}

# 0: string; 1: woodwind; 2: brass; 3: keyboard; 4: human voice
family_to_chr = {0: 'string', 1: 'woodwind', 2: 'brass', 3: 'keyboard', 4: 'human_voice'}
instrument_family = {
   'gel': 0,
   'gac': 0,
   'cel': 0,
   'vio': 0,
   'cla': 1,
   'sax': 1,
   'flu': 1,
   'tru': 2,
   'org': 3,
   'pia': 3,
   'voi': 4,
}

# map instruments to integer
label_dict = {
   'cel': 0,
   'cla': 1,
   'flu': 2,
   'gac': 3,
   'gel': 4,
   'org': 5,
   'pia': 6,
   'sax': 7,
   'tru': 8,
   'vio': 9,
   'voi': 10
}


def write_train_manifest(path,metadata_path):
    train_manifest = {}
    valid_manifest = {}
    for label in label_dict.keys():
        datalist = glob.glob(os.path.join(path, label, '**/*.wav'), recursive=True)
        songbook = {}
        for p in datalist:
            fullname = p.split('/')[-1]
            basename = fullname[:-4]
            songname = fullname[:-7]
            if songname not in songbook:
                songbook[songname] = [basename]
            else:
                songbook[songname].append(basename)

        # train test 0.85:0.15
        songlist = list(songbook.keys())
        train_list, valid_list = train_test_split(songlist, test_size=0.15, random_state=730)

        # write train metadata
        for tr_song in train_list:
            for clip in songbook[tr_song]:
                # update metadata
                if clip in train_manifest:
                    print('what!')
                    exit()
                relative_path = os.path.join(label, clip)
                train_manifest[clip] = {
                    'sample_rate': 44100,
                    'instrument_str': label,
                    'instrument': label_dict[label],
                    'onset_class_str': onset_to_chr[onset_group[label]],
                    'onset_class': onset_group[label],
                    'instrument_family_str': family_to_chr[instrument_family[label]],
                    'instrument_family': instrument_family[label],
                    'note_str': clip,
                    'relative_path': relative_path,
                    'song_name': tr_song
                }

        # write valid metadata
        for va_song in valid_list:
            for clip in songbook[va_song]:
                # update metadata
                relative_path = os.path.join(label, clip)
                valid_manifest[clip] = {
                    'sample_rate': 44100,
                    'instrument_str': label,
                    'instrument': label_dict[label],
                    'onset_class_str': onset_to_chr[onset_group[label]],
                    'onset_class': onset_group[label],
                    'instrument_family_str': family_to_chr[instrument_family[label]],
                    'instrument_family': instrument_family[label],
                    'note_str': clip,
                    'relative_path': relative_path,
                    'song_name': va_song
                }

    print('train set length: {}'.format(len(train_manifest)))
    print('valid set length: {}'.format(len(valid_manifest)))

    with open(os.path.join(metadata_path,'irmas_train.json'), 'w') as f:
        json.dump(train_manifest, f, indent=4)

    with open(os.path.join(metadata_path,'irmas_valid.json'), 'w') as g:
        json.dump(valid_manifest, g, indent=4)


def write_test_metadata(path,metadata_path):
    test_manifest = {}
    datalist = glob.glob(os.path.join(path, '**/*.wav'), recursive=True)
    for p in datalist:
        # for testing data, labels should be lists.
        labels_chr = []
        labels = []
        onsets_chr = []
        onsets = []
        families_chr = []
        families = []

        annotation_file = p[:-4] + '.txt'
        with open(annotation_file, 'r') as f:
            for line in f.readlines():
                label = line.strip()
                labels_chr.append(label)
                labels.append(int(label_dict[label]))

                onsets_chr.append(onset_to_chr[onset_group[label]])
                onsets.append(onset_group[label])

                families_chr.append(family_to_chr[instrument_family[label]])
                families.append(instrument_family[label])

        fullname = p.split('/')[-1]
        partname = p.split('/')[-2]
        basename = fullname[:-4]
        relative_path = os.path.join(partname, basename)

        test_manifest[basename] = {
            'sample_rate': 44100,
            'instrument_str': labels_chr,
            'instrument': labels,
            'onset_class_str': onsets_chr,
            'onset_class': onsets,
            'instrument_family_str': families_chr,
            'instrument_family': families,
            'note_str': basename,
            'relative_path': relative_path,
            'song_name': basename
        }

    print('test set length: {}'.format(len(test_manifest)))

    with open(os.path.join(metadata_path,'irmas_test.json'), 'w') as f:
        json.dump(test_manifest, f, indent=4)

def slice():
    # divide the audio into 1s clips for train/val split.
    # test set is divided on-the-fly when testing.
    meta = {}
    slice_meta = defaultdict(dict)
    n_slice = 3

    for split in ['train', 'valid']:
        with open(f'metadata/irmas_{split}.json') as f:
            meta[f'{split}'] = json.load(f)
            for k, v in meta[f'{split}'].items():
                for idx in range(n_slice):
                    s = k + f'_{idx}'
                    slice_meta[f'{split}'][s] = v.copy()
                    slice_meta[f'{split}'][s]['start'] = idx
                    slice_meta[f'{split}'][s]['end'] = idx + 1

    print('{} train samples, {} valid samples'.format(len(slice_meta['train']), len(slice_meta['valid'])))

    with open(os.path.join('metadata', 'irmas_slice_train.json'), 'w') as gg:
        json.dump(slice_meta['train'], gg, indent=4)

    with open(os.path.join('metadata', 'irmas_slice_valid.json'), 'w') as bb:
        json.dump(slice_meta['valid'], bb, indent=4)

if __name__ == '__main__':
    write_train_manifest(path='irmas_data/IRMAS-TrainingData',metadata_path='metadata')
    write_test_metadata(path='irmas_data/IRMAS-TestingData',metadata_path='metadata')
    slice()