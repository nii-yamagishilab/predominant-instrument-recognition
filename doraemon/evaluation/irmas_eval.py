from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from musyn.data.utils import lable_to_enc_dict

def evaluate_model_on_irmas(preds, labels, threshold_list=[0.5], one_hot=True, mode='f1'):
    res = pd.DataFrame()
    # preds = np.array(preds)
    # preds.numpy()
    if mode == 'acc':
        prediction = []
        target = []
        for pred in preds:
            prediction_sub = []
            for t in pred.flatten():
                prediction_sub.append(t)
            prediction += sorted(prediction_sub)

        for label in labels:
            target_sub = []
            print(label)
            for k in range(len(label)):
                if int(label[k]) == 1:
                    target_sub.append(k)
            target += sorted(target_sub)
        prediction = np.array(prediction)
        target = np.array(target)
        acc = (prediction==target).sum() / prediction.shape[0]
        return 'accuracy: {} %'.format(round(acc,4) * 100)


    if one_hot is False:
        INSTR = [*range(11)]
        Instr_DICT_ONEHOT = lable_to_enc_dict(INSTR, True)
        print(Instr_DICT_ONEHOT)
        converted = []
        for sample in preds:
            one_hot_out = np.zeros(11,dtype='float64')
            for t in sample:
                one_hot_out += Instr_DICT_ONEHOT[int(t)]
            converted.append(one_hot_out)
        res = compute_score(None,converted,labels)
        print(res)
    else:
        # apply threshold
        for thres in threshold_list:
            final_preds = apply_threshold(thres,preds)
            # compute metrics
            eval_result = compute_score(thres, final_preds, labels)
            res = res.append(eval_result, ignore_index=True)
    return res

def apply_threshold(th, score):
    score = np.array(score)
    score[score >= th] = 1
    score[score < th] = 0
    return score

def compute_score(threshold, preds, labels):
    metrics = {'threshold': "{}".format(threshold)}
    for avg in ['micro', 'macro']:
        f1 = f1_score(labels, preds, average=avg)
        p = precision_score(labels, preds, average=avg)
        r = recall_score(labels, preds, average=avg)
        metrics.update(
            {
                'precision_{}'.format(avg): round(p, 3),
                'recall_{}'.format(avg): round(r,3),
                'f1_score_{}'.format(avg): round(f1,3)
            }
        )
    return metrics

if __name__ == '__main__':
    preds = torch.Tensor([[1,2],[3,2]])
    labels = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
    eval_result = evaluate_model_on_irmas(preds,labels,one_hot=False)
    print(eval_result)