import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from itertools import compress

label_dict = {
    'cel': 0,
    'cla': 1,
    'flu': 2,
    'a-gt': 3,
    'e-gt': 4,
    'organ': 5,
    'piano': 6,
    'sax': 7,
    'tru': 8,
    'violin': 9,
    'voice': 10
}

label_dict_full = {
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
# d_label = {v:k for k,v in label_dict.items()}


def plot_prediction_map(preds_map, targets, note, save_dir, dpi=250):
    preds_map = preds_map.transpose(0,1).detach().cpu()
    mean_preds = torch.mean(preds_map, dim=1, keepdim=True)
    final_preds = apply_threshold(mean_preds, 0.12)

    targets = targets.squeeze().long()
    targets_name = list(compress(list(label_dict.keys()), list(map(bool, targets.tolist()))))
    targets = (targets == 1).nonzero(as_tuple=True)[0].long().tolist()
    tt_name = ', '.join(targets_name)
    tt = ', '.join(list(map(str, targets)))

    preds_labels = final_preds.squeeze(1).long()
    preds_labels_name = list(compress(list(label_dict.keys()), list(map(bool, preds_labels.tolist()))))
    pred_labels = (preds_labels == 1).nonzero(as_tuple=True)[0].long().tolist()
    pp_name = ', '.join(preds_labels_name)
    pp = ', '.join(list(map(str, pred_labels)))



    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True,gridspec_kw={'width_ratios': [5, 1, 1]})

    hm = pd.DataFrame(data=preds_map, index=list(label_dict_full))
    sns.heatmap(ax=axes[0], data=hm, square=True, cmap="YlGnBu")

    hm_2 = pd.DataFrame(data=mean_preds, index=list(label_dict_full))
    sns.heatmap(ax=axes[1], data=hm_2, square=True, cmap="YlGnBu")

    hm_3 = pd.DataFrame(data=final_preds, index=list(label_dict_full))
    sns.heatmap(ax=axes[2], data=hm_3, square=True, cmap="YlGnBu")

    axes[0].set_xlabel('Time (Frame)')
    axes[0].set_ylabel('Logits')

    axes[1].set_title('mean')

    axes[2].set_title('th=0.12')

    plt.figtext(.55, .03, "Preds: {} / {} ".format(pp_name, pp), style='italic', color='black', bbox={
        'facecolor': 'none', 'edgecolor' : 'green', 'alpha': 0.5, 'pad': 6})
    plt.figtext(.8, .03, "Targets: {} / {} ".format(tt_name,tt), style='italic', color='black', bbox={
        'facecolor': 'none', 'edgecolor' : 'blue', 'alpha': 0.5, 'pad': 6})

    fig.suptitle('{} - Prediction Maps'.format(note))
    plt.savefig(os.path.join(save_dir, '{}_prediction_maps.png'.format(note)), dpi=dpi)
    plt.close()


def apply_threshold(preds, threshold):
    preds = torch.where(preds >= threshold, 1, 0)
    return preds