# adapted from: https://github.com/Alexuan/musical_instrument_embedding by Lifan Zhong
import torch.nn as nn

from src.models.mie.backbones import resnet34, se_resnet34, resnet50, visual_resnet50, resnet50_e1, shared_resnet34, v_resnet34
from src.models.mie.heads import LinearClsHead, AngularClsHead
from src.models.mie.necks import LDE, SAP
from src.models.mie.transforms import SincConv


class InstrEmdSincModel(nn.Module):
    """Embedding Net (Instrument Embedding SincConv Model) 
    """

    def __init__(self, opt):
        super(InstrEmdSincModel, self).__init__()
        self.opt = opt
        # tranfer raw audio to sinc feature
        kernel_size = int(opt.transform.kernel_size / 1000 * opt.transform.sr)
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        stride = int(opt.transform.stride / 1000 * opt.transform.sr)
        self.trans = SincConv(
            out_channels=opt.transform.out_channels,
            kernel_size=kernel_size,
            in_channels=opt.transform.in_channels,
            padding=opt.transform.padding,
            stride=stride,
            init_type=opt.transform.init_type,
            min_low_hz=opt.transform.min_low_hz,
            min_band_hz=opt.transform.min_band_hz,
            requires_grad=opt.transform.requires_grad,
        )

        # norm
        if opt.spec_bn:
            self.spec_bn = nn.BatchNorm2d(1)
            print('spec bn bn bn bn')
        else:
            self.spec_bn = nn.Identity()
            print('spec bn no no no')

        # encode sinc feature to latent space
        if opt.backbone.type == 'se_resnet34':
            self.encoder = se_resnet34()
            _feature_dim = 128
        elif opt.backbone.type == 'resnet34':
            self.encoder = resnet34()
            _feature_dim = 128
        elif opt.backbone.type.startswith('shared_resnet34'):
            layer_nums_txt = opt.backbone.type[len('shared_resnet34') + 1:]
            self.encoder = shared_resnet34(layer_nums_txt)
            _feature_dim = 128
        elif opt.backbone.type.startswith('v_resnet34'):
            layer_nums_txt = opt.backbone.type[len('v_resnet34') + 1:]
            self.encoder = v_resnet34(layer_nums_txt)
            _feature_dim = 128
            if int(layer_nums_txt[-1:]) == 8:
                _feature_dim = 64
            if int(layer_nums_txt[-2:]) == 32:
                _feature_dim = 256
            if int(layer_nums_txt[-2:]) == 64:
                _feature_dim = 512

        elif opt.backbone.type == 'resnet50':
            self.encoder = resnet50()
            _feature_dim = 128 * 2
        elif opt.backbone.type == 'visual_resnet50':
            self.encoder = visual_resnet50()
            _feature_dim = 2048
        elif opt.backbone.type == 'resnet50_e1':
            self.encoder = resnet50_e1()
            _feature_dim = 128
        else:
            raise NotImplementedError

        # backbone pretrain removed

        if opt.neck.type == 'LDE':
            self.pool = LDE(
                D=opt.neck.D,
                input_dim=_feature_dim,
                pooling=opt.neck.pooling,
                network_type=opt.neck.network_type,
                distance_type=opt.neck.distance_type
            )

            if opt.neck.pooling == 'mean':
                in_channels = _feature_dim * opt.neck.D
            if opt.neck.pooling == 'mean+std':
                in_channels = _feature_dim * 2 * opt.neck.D
        elif opt.neck.type == 'SAP':
            self.pool = SAP(
                dim=_feature_dim,
                n_heads=opt.neck.n_heads,
            )
            in_channels = _feature_dim * opt.neck.n_heads
        else:
            raise NotImplementedError

        self.fc0 = nn.Linear(in_channels, opt.head1.hidden_dim)
        ############# Glorot, X. & Bengio, Y. initialization
        # nn.init.xavier_normal_(self.fc0.weight)
        #############
        self.bn0 = nn.BatchNorm1d(opt.head1.hidden_dim)

        if opt.head1.type == 'AngularClsHead':
            self.fc11 = AngularClsHead(
                num_classes=opt.head1.num_classes,
                in_channels=opt.head1.hidden_dim,
                m=opt.head1.m,
            )
        elif opt.head1.type == 'LinearClsHead':
            self.fc11 = LinearClsHead(
                num_classes=opt.head1.num_classes,
                in_channels=opt.head1.hidden_dim,
            )
        else:
            raise NotImplementedError

        # multi-task
        # try:
        #     if opt.head2:
        #         self.fc12 = LinearClsHead(
        #             num_classes=opt.head2.num_classes,
        #             in_channels=opt.head2.hidden_dim,
        #         )
        #         self.is_mt = True
        #         print('[MODEL] running multi-task!')
        # except:
        #     self.is_mt = False
        #     print('[MODEL] runing single-task!')

    def forward(self, x):
        # x = x.transpose(1, -1)  # batch * time * channel
        x = self.trans(x)

        x = x.unsqueeze(1)  # --> torch.Size([128, 1, 250, 122])

        x = self.spec_bn(x)

        ####
        embs = self.encoder(x)
        x = self.pool(embs)
        ####

        feat = self.fc0(x)
        feat_bn = self.bn0(feat)

        out = self.fc11(feat_bn)
        # if self.is_mt:
        #     out_2 = self.fc12(feat_bn)
        #     return feat, (out, out_2)

        return feat, out

    def predict(self, x):
        x = x.transpose(1, -1)  # batch * time * channel
        x = self.trans(x)
        x = self.encoder(x)
        x = self.pool(x)
        if type(x) is tuple:
            x = x[0]
        feat = self.fc0(x)
        return feat
