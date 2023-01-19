import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import plot_det_curve
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def plot_TSNE(feat, labels, save_dir, note, dpi=250, cmap='rainbow'):
   # tsne = TSNE(n_components=2, verbose=1, perplexity=30, learning_rate=50.0, n_iter=500)
   tsne = TSNE(n_components=2, random_state=1103)
   feat_embedded = tsne.fit_transform(feat)

   font = {'size': 25}
   matplotlib.rc('font', **font)
   fig = plt.figure(figsize=(18, 10))
   ax = fig.add_subplot(1, 1, 1)

   tx = scale_to_01_range(feat_embedded[:, 0])
   ty = scale_to_01_range(feat_embedded[:, 1])

   label_dict = {
       'cello': 0,
       'clarinet': 1,
       'flute': 2,
       'acoustic guitar': 3,
       'electric guitar': 4,
       'organ': 5,
       'piano': 6,
       'saxophone': 7,
       'trumpet': 8,
       'violin': 9,
       'voice': 10
   }

   color_dict = list(sns.color_palette(cmap, 11))
   label_dict_inverse = {v: k for k, v in label_dict.items()}
   labels_A_str = np.vectorize(label_dict_inverse.get)(labels)

   for label_idx in range(11):
       # find the samples of the current class in the data
       indices = [i for i, l in enumerate(labels) if l == label_idx]

       # extract the coordinates of the points of this class only
       current_tx = np.take(tx, indices)
       current_ty = np.take(ty, indices)

       # convert the class color to matplotlib format
       color = np.repeat(np.atleast_2d(list(color_dict[label_idx])), current_tx.shape[0], axis=0)

       # add a scatter plot with the corresponding color and label
       ax.scatter(current_tx, current_ty, c=color, label=label_dict_inverse[label_idx], cmap='rainbow')
   ax.legend(loc='upper right', markerscale=2.0, bbox_to_anchor=(1.25, 1.))
   fig.tight_layout()
   fig.savefig(os.path.join(save_dir, 't-SNE-{}.jpg'.format(note)), dpi=dpi)
   plt.close()


def plot_TSNE_before_LDE(feat, save_dir, prefix=None):
   tsne = TSNE(n_components=2, verbose=1, perplexity=20, learning_rate=15.0, n_iter=250)
   feat_embedded = tsne.fit_transform(feat)

   font = {'size': 25}
   matplotlib.rc('font', **font)
   fig = plt.figure(figsize=(18, 10))
   ax = fig.add_subplot(1, 1, 1)

   tx = scale_to_01_range(feat_embedded[:, 0])
   ty = scale_to_01_range(feat_embedded[:, 1])

   label_dict = {
       'cello': 0,
       'clarinet': 1,
       'flute': 2,
       'acoustic guitar': 3,
       'electric guitar': 4,
       'organ': 5,
       'piano': 6,
       'saxophone': 7,
       'trumpet': 8,
       'violin': 9,
       'voice': 10
   }

   ax.scatter(x=tx,
              y=ty)
   # # ax.legend(*scatter.legend_elements(), title=prefix)
   ax.legend(loc='upper right', markerscale=2.0, bbox_to_anchor=(1.25, 1.))

   fig.tight_layout()
   fig.savefig(os.path.join(save_dir, 't-SNE-LDE-{}.jpg'.format(prefix)), dpi=600)


def scale_to_01_range(x):
   # compute the distribution range
   value_range = (np.max(x) - np.min(x))

   # move the distribution so that it starts from zero
   # by extracting the minimal value from all its values
   starts_from_zero = x - np.min(x)

   # make the distribution fit [0; 1] by dividing by its range
   return starts_from_zero / value_range


def plot_DET(x, y, eer, save_dir):
   fig = plt.figure()
   plt.title('DET')
   plt.xlabel('False Positives')
   plt.ylabel('True Positive rate')
   plt.plot(x, y, label='EER: ' + str(eer))
   fig.savefig(os.path.join(save_dir, 'DET.png'), dpi=600)

