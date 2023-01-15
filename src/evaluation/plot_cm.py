import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

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

def plot_confusion_matrix(labels, preds, note, save_dir,dpi=250, normalize=False):

   if normalize:
       cm = np.round(confusion_matrix(labels, preds, normalize='true'), 2)
   else:
       cm = confusion_matrix(labels, preds)
   # plt.rcParams["figure.figsize"] = (9,10)
   cm = pd.DataFrame(data=cm, index=list(label_dict),
                     columns=list(label_dict))
   sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues', annot_kws={"size": 6.5}, )
   plt.xlabel('Predicted Labels')
   plt.ylabel('Ground Truth')
   if normalize:
       plt.title('{} - Normalized CM'.format(note))
       plt.savefig(os.path.join(save_dir, '{}_normalized_cm.png'.format(note)), dpi=dpi)
   else:
       plt.title('{} - CM'.format(note))
       plt.savefig(os.path.join(save_dir, '{}_cm.png'.format(note)), dpi=dpi)
   plt.close()
