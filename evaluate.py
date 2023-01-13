import argparse
import os.path
import glob

from doraemon import Evaluator
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
