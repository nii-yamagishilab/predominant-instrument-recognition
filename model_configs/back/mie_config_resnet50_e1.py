model = dict(
    type='instr_emd_sinc',
    transform=dict(
        type='SincConv',
        sr=16000,
        out_channels=122,
        kernel_size=50,
        stride=12,
        in_channels=1,
        padding='same',
        init_type='midi',
        min_low_hz=5,
        min_band_hz=5,
        requires_grad=True,
    ),
    backbone=dict(
        type='resnet50_e1',
        pretrained='',
    ),
    # neck=dict(
    #     type='SAP',
    #     n_heads=1,
    # ),
    neck=dict(
        type='LDE',
        D=8,
        pooling='mean',
        network_type='lde',
        distance_type='sqr'
    ),
    head1=dict(
        type='LinearClsHead',
        num_classes=1006,
        hidden_dim=512,
    ),
)