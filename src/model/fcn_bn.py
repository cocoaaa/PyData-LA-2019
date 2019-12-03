import functools

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels


from helpers import random_seed
import pdb
# from ptsemseg.models.utils import get_upsampling_weight
# from ptsemseg.loss import cross_entropy2d


# FCN32s
class fcn32_bn(nn.Module):
    def __init__(self, in_channels=3, n_classes=21, learned_billinear=False):
        super(fcn32s, self).__init__()
        self.in_channels = in_channels
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.name = 'fcn32_bn'
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),
        )

        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)

        out = F.upsample(score, x.size()[-2:])

        return out

    def init_vgg16_params(self, vgg16_bn, copy_bns, copy_fc8=False):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 6], [7, 13], [14, 23], [24, 33], [34, 43]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                print(f'vgg layer type: ', type(l1))
                print(f'fcn layer type: ', type(l2))
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
                elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d) and copy_bns:
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
                    print('Copied bn layer')
                    
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


class fcn16_bn(nn.Module):
    def __init__(self, in_channels=3, n_classes=21, learned_billinear=False):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.learned_billinear = learned_billinear
        self.name = 'fcn16_bn'
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
#             nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, 1),
#             nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, self.n_classes, 1),
        )

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        avgpooled = self.avgpool(conv5)
        
        score = self.classifier(avgpooled)
        score_pool4 = self.score_pool4(conv4)

        score = F.upsample(score, score_pool4.size()[-2:])
        score += score_pool4
        out = F.upsample(score, x.size()[-2:])
        return out
    
    def init_vgg16_bn_params(self, vgg16_bn, copy_bns, copy_classifier, copy_last_classifier=True, verbose=False):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 6], [7, 13], [14, 23], [24, 33], [34, 43]]
        features = list(vgg16_bn.features.children())

        for idx, conv_block in enumerate(blocks):
            if verbose:
                print(f'\nfeatrue range: {ranges[idx]}')
                print(f'conv-block: {conv_block}')
            print(f'Copying to conv_block{idx+1}')
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
                    print('\tCopied conv layer')
                    
                elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d) and copy_bns:
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
                    print('\tCopied bn layer')
        if copy_classifier:            
            for i1, i2 in zip([0, 3], [0, 3]):
                l1 = vgg16_bn.classifier[i1]
                l2 = self.classifier[i2]
                l2.weight.data = l1.weight.data.view(l2.weight.size())
                l2.bias.data = l1.bias.data.view(l2.bias.size())
                print('Copied first two classifier layers')

            if copy_last_classifier:
                l1 = vgg16_bn.classifier[-1]
                l2 = self.classifier[-1]            
                l2.weight.data = l1.weight.data[:self.n_classes, :].view(l2.weight.size())
                l2.bias.data = l1.bias.data[:self.n_classes]
                print('Copied last classifier layer')



# FCN 8s
class fcn8_bn(nn.Module):
    def __init__(self, n_classes=21, learned_billinear=True):
        super(fcn8s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.name = 'fcn8_bn'
#         self.loss = functools.partial(cross_entropy2d, size_average=False)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),
        )

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

        if self.learned_billinear:
            self.upscore2 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 4, stride=2, bias=False
            )
            self.upscore4 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 4, stride=2, bias=False
            )
            self.upscore8 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 16, stride=8, bias=False
            )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(
                    get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                )

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)

        if self.learned_billinear:
            upscore2 = self.upscore2(score)
            score_pool4c = self.score_pool4(conv4)[
                :, :, 5 : 5 + upscore2.size()[2], 5 : 5 + upscore2.size()[3]
            ]
            upscore_pool4 = self.upscore4(upscore2 + score_pool4c)

            score_pool3c = self.score_pool3(conv3)[
                :, :, 9 : 9 + upscore_pool4.size()[2], 9 : 9 + upscore_pool4.size()[3]
            ]

            out = self.upscore8(score_pool3c + upscore_pool4)[
                :, :, 31 : 31 + x.size()[2], 31 : 31 + x.size()[3]
            ]
            return out.contiguous()

        else:
            score_pool4 = self.score_pool4(conv4)
            score_pool3 = self.score_pool3(conv3)
            score = F.upsample(score, score_pool4.size()[2:])
            score += score_pool4
            score = F.upsample(score, score_pool3.size()[2:])
            score += score_pool3
            out = F.upsample(score, x.size()[2:])

        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]

            
# Get a new model -- reproducible with fixed seeds
def get_fcn16_bn(device, seed, pretrain, copy_bns, copy_classifier, **kwargs):
    """
    - kwargs:
    
    1. Below keywords will be passed to fcn16_bn class constructor
        - in_channels
        - n_channels
        - learned_billinear # DEBUG : is this original type?
    2. These keywords will be passed to fcn15_fn's init_vgg16_bn_params method  
        - Matters only when pretrain is True
        - copy_bns (bool, mandatory argument)
        - copy_last_classifier (default: True)
        - verbose (bool, default: False)
    """
    use_cuda = False if device =='cpu' else True
    random_seed(seed, use_cuda)
    
    # Get parameters
    in_channels = kwargs.get('in_channels', 3)
    n_classes = kwargs.get('n_classes', 21)
    learned_billinear = kwargs.get('learned_billinear', False)
    
    model = fcn16_bn(in_channels=in_channels, n_classes=n_classes, 
                     learned_billinear=learned_billinear)
    
    if pretrain:
        # Get initialization kwargs
#         copy_classifier 
        copy_last_classifier = kwargs.get('copy_last_classifier', True)
        verbose = kwargs.get('verbose', False)
        
        vgg16_bn = tvmodels.vgg16_bn(pretrained=True)
        model.init_vgg16_bn_params(vgg16_bn, 
                                   copy_bns, 
                                   copy_classifier, 
                                   copy_last_classifier=copy_last_classifier,
                                   verbose=verbose)
    model.name = f'fcn16_bn-pretrain-{pretrain}-copy_bns-{copy_bns}-copy_classifier-{copy_classifier}'
    model = model.to(device)
    return model

## Usage
# model = get_fcn16_bn(DEVICE, seed=1, pretrain=True, copy_bns=True)
# print(model)
# summary(model, (3,244,244))