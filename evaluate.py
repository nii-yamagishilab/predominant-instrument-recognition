import argparse
import os.path
import glob

import numpy as np
from src import Evaluator
import warnings

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--model_name', type=str, default='irmas_mie')
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--act', type=str, default='sigmoid')
    parser.add_argument('--pp', type=str, default='average')
    parser.add_argument('--tem', type=float, default=1.0)
    parser.add_argument('--seg_dur', type=int, default=3)
    parser.add_argument('--add_up', action='store_true')
    parser.add_argument('--no-add_up', dest='add_up', action='store_false')
    parser.set_defaults(add_up=False)

    args = parser.parse_args()
    return args


def enc_pro(config):
    import json
    import torchaudio
    from tqdm import tqdm
    testing_data = json.load(open(config['test_meta_path'], 'r'))
    print(len(testing_data.keys()))
    for k, v in tqdm(testing_data.items()):
        song = os.path.join(config['test_data_dir'], v['relative_path'] + '.wav')
        s, sr = torchaudio.load(song)


if __name__ == '__main__':
    args = get_args()

    config = {
        'model_name': 'resnet50',
        'ckpt': '',
        'valid_meta_path': 'metadata/irmas_valid.json',
        'valid_data_dir': 'irmas_data/IRMAS-TrainingData/',
        'test_meta_path': 'metadata/irmas_test.json',
        'test_data_dir': 'irmas_data/IRMAS-TestingData/',
    }
    if args.model_name:
        config['model_name'] = args.model_name

    if args.mode == 'report':
        import statistics
        import pandas as pd

        print("---------Making Reports----------")
        results = {}
        avg_results = {}
        if args.model_name == 'irmas_mie':
            lr = '2.5e-4'
        elif args.model_name == 'no_pre_irmas_mie':
            lr = '3.5e-3'
        for seed_n in [1231, 2233, 906]:
            model_summary = True if seed_n == 1231 else False
            dir_name = 'seed-{}-lr-{}-bs-64-epoch-40-warmup-5'.format(seed_n, lr)
            if args.model_name == 'no_pre_irmas_mie':
                dir_name += '-relu'
            ckpt = os.path.join(args.ckpt_dir, dir_name, 'checkpoints', 'last.ckpt')
            print('currently evaluating: {}'.format(ckpt))
            config['ckpt'] = ckpt
            e = Evaluator(**config)
            results['seed-{}'.format(seed_n)] = e.test(activation=args.act,
                                                       pp=args.pp,
                                                       tem=args.tem,
                                                       seg_dur=args.seg_dur,
                                                       add_up=args.add_up,
                                                       model_summary=model_summary,
                                                       )
            print(results['seed-{}'.format(seed_n)]['iw'])
        iw_lst = []
        iw_results = {}
        f1_micro_lst = []
        f1_macro_lst = []
        lrap_lst = []
        for k, v in results.items():
            f1_micro_lst.append(float(v['best_f1_micro']))
            f1_macro_lst.append(float(v['best_f1_macro']))
            lrap_lst.append(float(v['LRAP']))
            iw_lst.append(v['iw'])
            iw_results[k] = v['iw']

        avg_f1_micro = round(statistics.mean(f1_micro_lst), 3)
        std_f1_micro = round(statistics.stdev(f1_micro_lst), 4)

        avg_f1_macro = round(statistics.mean(f1_macro_lst), 3)
        std_f1_macro = round(statistics.stdev(f1_macro_lst), 4)

        avg_lrap = round(statistics.mean(lrap_lst), 3)
        std_lrap = round(statistics.stdev(lrap_lst), 4)

        avg_iw = np.mean(iw_lst, 0)
        iw_lst.append(avg_iw)

        reports = {
            'f1_micro': [avg_f1_micro, std_f1_micro],
            'f1_macro': [avg_f1_macro, std_f1_macro],
            'LRAP': [avg_lrap, std_lrap],
        }
        df0 = pd.DataFrame.from_dict(reports)
        df0.to_csv(os.path.join(args.ckpt_dir, 'test-mean+std.csv'))
        print(reports)

        iw_results['average'] = np.around(iw_lst[-1], decimals=3)
        df = pd.DataFrame.from_dict(iw_results)
        df.to_csv(os.path.join(args.ckpt_dir, 'instrument-wise-analysis.csv'))
        exit()

    if args.ckpt_dir:
        ckpts = glob.glob(os.path.join(args.ckpt_dir, '**/*.ckpt'), recursive=True)
        for ckpt in ckpts:
            print('currently evaluating: {}'.format(ckpt))
            config['ckpt'] = ckpt
            e = Evaluator(**config)
            if args.mode == 'eval':
                e.evaluate_model()
            elif args.mode == 'plot':
                e.plot()
            elif args.mode == 'test':
                e.test(activation=args.act,
                       pp=args.pp,
                       tem=args.tem,
                       seg_dur=args.seg_dur,
                       add_up=args.add_up,
                       )
            else:
                raise NotImplementedError

        exit()

    elif args.ckpt:
        config['ckpt'] = args.ckpt

    e = Evaluator(**config)
    if args.mode == 'eval':
        e.evaluate_model()
    elif args.mode == 'plot':
        e.plot()
    elif args.mode == 'test':
        e.test(activation=args.act,
               pp=args.pp,
               tem=args.tem,
               seg_dur=args.seg_dur,
               add_up=args.add_up,
               )
    elif args.mode == 'plot_pred':
        e.plot_pred_map()
    else:
        raise NotImplementedError
