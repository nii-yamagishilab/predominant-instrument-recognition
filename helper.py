# from doraemon.models.resnet50 import IRResNet50
import glob
import os.path

from tqdm import tqdm
from src.models.mie.backbones import resnet34, se_resnet34, resnet50, visual_resnet50, shared_resnet34
# from doraemon.models.instr_emd_sinc_model import InstrEmdSincModel
# from doraemon.utils import Config
import torch
import sys
# from doraemon.datasets.irmas_dataset import IRMASDataset
# from doraemon.datasets.nsynth_dataset import NSynthDataset
import torch.nn as nn
from torchaudio_augmentations import *
from src.lms.mie import InstrumentRecognizer
from torchsummary import summary


def load_model(cfg_path):
    cfg = Config.fromfile(cfg_path)
    m = InstrEmdSincModel(opt=cfg.model)
    print(m)


def check_data():
    data = NSynthDataset(meta_path='metadata/nsynth_train.json', wav_dir='nsynth_data', augment='stitch')
    data[0]
    # test_data = IRMASDataset(meta_path='metadata/irmas_test.json',wav_dir='irmas_data/IRMAS-TestingData',is_test=True)
    # print(len(data))


def check_bce():
    l = nn.BCELoss()
    logsfm = nn.LogSoftmax(dim=1)
    nllloss = nn.NLLLoss()
    ll = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    a = torch.Tensor([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 1., 0.]])
    b = torch.Tensor([[0., 1., 1.], [1., 0., 1.], [0., 1., 1.], [1., 0., 1.]])
    model_out = torch.Tensor([995.4628, 995.3303, 994.3607, 995.4346, 996.3753, 994.4213, 994.8397,
                              995.4814, 996.0850, 995.3153, 995.1248]).unsqueeze(0)
    sgm_out = torch.sigmoid(model_out)
    sfm_out = logsfm(model_out)
    gt = torch.Tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).unsqueeze(0)
    # print(sgm_out)
    # print(sfm_out)

    # print(ce(model_out, gt))
    print(nllloss(sfm_out, torch.argmax(gt).unsqueeze(0)))
    print(l(sfm_out, gt))
    print(torch.nn.functional.binary_cross_entropy(sfm_out, gt))
    print(ll(model_out, gt))


def nl_loss(input, target):
    return -input[range(target.shape[0]), target].log().mean()


def check_ckpt():
    ckpt_xuan = torch.load('nsynth_weights.h5')
    print(ckpt_xuan['state_dict'].keys())
    # ckpt_new = torch.load('/home/smg/v-zhonglifan/InstrumentRecognition/exp16-mie-pretrain/tb_logs/mie-34-aug-CE-pretraining-cosine-adamw/lr-1e-3-seed-2233/checkpoints/epoch=21-valid_acc=0.906.ckpt')
    # re_name_ckpt = {}
    # for k,v in ckpt_new['state_dict'].items():
    #     re_name_ckpt[k[6:]] = v
    # model_dir = "new_train.h5"
    # torch.save({
    #     'state_dict': re_name_ckpt
    # },
    #     model_dir
    # )


def check_resnet():
    a = torch.randn(2,1,250,80).cuda()
    m = resnet34().cuda()
    o = m(a)
    print(o.shape)
    # summary(shared_resnet34().cuda(), input_size=(1, 250, 128))


def augments():
    song = torch.randn(12, 1, 16000)
    transforms = [
        RandomApply([PolarityInversion()], p=0.8),
        RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
        RandomApply([Gain()], p=0.2),
        HighLowPass(sample_rate=16000),  # this augmentation will always be applied in this aumgentation chain!
        RandomApply([Delay(sample_rate=16000)], p=0.5),
        RandomApply([PitchShift(
            n_samples=1,
            sample_rate=16000
        )], p=0.4),
        RandomApply([Reverb(sample_rate=16000)], p=0.3)
    ]
    transform = Compose(transforms=transforms)
    o = transform(song)
    print(o.shape)


def set_model():
    feature_extractor = InstrumentRecognizer.load_from_checkpoint(
        '/home/smg/v-zhonglifan/InstrumentRecognition/exp18-mie-modify/tb_logs/mie-aug-34-trim-CE-pretraining-cosine-adam-seg_dur-1/lr-1e-3-seed-2233-mixup_alpha-0.4-ls-0.05/checkpoints/last.ckpt')
    feature_extractor = feature_extractor.model
    feature_extractor.fc0 = nn.Identity()
    feature_extractor.bn0 = nn.Identity()
    feature_extractor.fc11 = nn.Identity()
    # feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-3])
    i = torch.randn(12, 16000)
    o = feature_extractor(i)
    print(o[0])
    print(o[1])
    # feature_extractor.fc11.fc = torch.nn.Linear(in_features=feature_extractor.fc11.fc.in_features, out_features=11)


def rename_project_file():
    file_lst = glob.glob(os.path.join('./','**/*.sh'),recursive=True)
    for file_path in tqdm(file_lst):
        with open(file_path, 'r') as file:
            filedata = file.read()

        # Replace the target string
        filedata = filedata.replace('doraemon', 'src')

        # Write the file out again
        with open(file_path, 'w') as file:
            file.write(filedata)

def rename_model_file():
    ckpt_path = '/home/smg/v-zhonglifan/InstrumentRecognition/c-IR/model_ckpt_tmp/irnet4nsynth.ckpt'
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_name = os.path.basename(ckpt_path)
    ckpt = torch.load(ckpt_path)
    ckpt['hyper_parameters']['result_folder'] = './irnet'
    # ckpt['hyper_parameters']['model_path'] = './src/lms/mie.py'
    # ckpt['hyper_parameters']['pretrained'] = './tb_logs' + ckpt['hyper_parameters']['pretrained'].strip().split('tb_logs')[1]
    torch.save(ckpt,os.path.join(ckpt_dir,ckpt_name))

if __name__ == '__main__':
    # load_model(cfg_path='mie_config.py')
    # check_bce()
    # check_resnet()
    # set_model()
    # x = torch.rand(4,5)
    # y = torch.randint(5, size=(4,))
    # print(x)
    # print(y)
    # print(x[range(4),y])
    # rename_project_file()
    rename_model_file()