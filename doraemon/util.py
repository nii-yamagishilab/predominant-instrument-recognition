import json

import numpy as np
import torch
from os import cpu_count
from multiprocessing import Pool
from itertools import repeat
import matplotlib.pyplot as plt

def binarize(label_tensor, num_classes=11):
    out_bi = torch.zeros(label_tensor.shape[0], num_classes, dtype=torch.float32)  # BS x N_CLASSES
    for i in range(label_tensor.shape[0]):
        label_int = label_tensor[i].item()
        out_bi[i][label_int] = 1
    return out_bi

def binarize_multilabel(label_tensor, num_classes=11, dtype=torch.float32):
    out_bi = torch.zeros(num_classes, dtype=dtype)
    for i in range(len(label_tensor)):
        label_int = label_tensor[i].item()
        out_bi[label_int] = 1
    return out_bi

def plot_hist(mel):
    mel = mel.detach().cpu().numpy().ravel()
    plt.hist(mel, color='blue', edgecolor='black',
             bins=20)
    plt.savefig('dist.jpg')
    plt.close()


class Mel2Img():
    def __init__(self,feat_dir, feat_note='1024-p4'):
        self.feat_dir = feat_dir
        self.split = 'train'
        self.feat_note = feat_note

    def process(self):
        pass

    def save_to_img(self,
                    save_dir,
                    note,
                    manifest_path,
                    cmap='jet'
                   ):
        manifest = pd.read_csv(manifest_path)
        filenames = manifest['filename'].tolist()
        idx = manifest['idx'].tolist()
        indice = range(len(filenames))

        out_path = save_dir + '/images_{}'.format(note)
        if os.path.isdir(out_path):
            print('[INFO]skipping {} set preprocessing'.format(self.split))
            return
        else:
            print('[INFO] preprocessing {} set'.format(self.split))
            os.makedirs(out_path)
            num_workers = int(0.8 * cpu_count())
            chunksize = max(1, int(len(indice) / num_workers))
            start = time()
            with Pool(num_workers) as p:
                p.starmap(self.plot_mel,
                          zip(indice,
                              repeat(idx),
                              repeat(filenames),
                              repeat(out_path),
                              repeat(cmap)),
                          chunksize)
            print(f'Duration of parallel processing (Pool): {time() - start:.2f} secs')

    def plot_mel(self,
                 index,
                 idx,
                 filenames,
                 out_path,
                 cmap):
        single_audio_path = filenames[index].split('/')[-1]
        if self.split == 'test':
            single_feature_path = os.path.join(
                '/home/zhong_lifan/data/IRMAS-Precomputed/IRMAS-TrainingData-Features/{}'.format(self.feat_note),
                single_audio_path + '.npy')
        else:
            single_feature_path = os.path.join(
                '/home/zhong_lifan/data/IRMAS-Precomputed/IRMAS-TestingData-Features/{}'.format(self.feat_note),
                single_audio_path + '.npy')
        audio_index = int(idx[index])
        mel = np.load(single_feature_path)
        for s_idx, start_index in enumerate(range(0, mel.shape[1] - 43 + 1, 22)):
            segment = mel[:, start_index:(start_index + 43)]
            segment_name = '{}_{}'.format(audio_index, s_idx)
            self.convert_mel_to_img(segment, out_path=out_path, name=segment_name, cmap=cmap)

    def convert_mel_to_img(self,
                           spec,
                           out_path,
                           name,
                           aspect="auto",
                           cmap='jet'):
        fig, axs = plt.subplots(1, 1)
        im = axs.imshow(spec, origin="lower", aspect=aspect, cmap=cmap)
        plt.axis('off')
        plt.savefig(os.path.join(out_path, '{}.jpg'.format(name)),
                    bbox_inches='tight',
                    pad_inches=0.005)
        plt.close()

def plot_spectrogram(specgram, out_path, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(specgram, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.savefig(os.path.join(out_path, '{}.jpg'.format(title)),)
    plt.close()

def get_label_weights(meta_path):
    # [328 429 380 539 644 579 615 530 490 493 661] train

    with open(meta_path,'r') as j:
        meta = json.load(j)

    c = np.zeros(11,dtype=int)
    for k, v in meta.items():
        c[v['instrument']] += 1

    c_weight = np.exp(c) / sum(np.exp(c))

    print(c)
    print(c_weight)

if __name__ == "__main__":
    get_label_weights(meta_path='./metadata/irmas_train.json')