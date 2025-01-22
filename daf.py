import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import modules.macnnet as macnnet


def Conv_Stage(input_dim, dim_list, bias=True, output_map=False):
    num_layers = len(dim_list)
    dim_list = [input_dim] + dim_list

    layers = []
    for i in range(num_layers):
        layer = nn.Sequential(
            nn.Conv2d(dim_list[i], dim_list[i + 1], kernel_size=3, bias=bias, padding=1),
            nn.BatchNorm2d(dim_list[i + 1]),
            nn.ReLU(inplace=True)
        )
        layers.append(layer)

    if output_map:
        layer = nn.Conv2d(dim_list[-1], 1, kernel_size=1)
        layers.append(layer)

    ## with padding, doesn't change the resolution
    return nn.Sequential(*layers)


def Conv_Stage2(input_dim, dim_list, bias=True, output_map=False):
    num_layers = len(dim_list)
    dim_list = [input_dim] + dim_list

    layers = []
    for i in range(num_layers):
        layer = nn.Sequential(
            nn.Conv2d(dim_list[i], dim_list[i + 1], kernel_size=3, bias=bias, padding=1),
            nn.BatchNorm2d(dim_list[i + 1]),
            nn.ReLU(inplace=True)
        )
        layers.append(layer)

    if output_map:
        layer = nn.Conv2d(dim_list[-1], 8, kernel_size=1)
        layers.append(layer)

    ## with padding, doesn't change the resolution
    return nn.Sequential(*layers)


class RB_riddnet(nn.Module):
    def __init__(self, cin=3):
        super(RB_riddnet, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1_b = Conv_Stage(cin, [8, 4, 16])
        self.conv2_b = nn.Sequential(
            nn.Conv2d(cin, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.soft_boundary = Conv_Stage2(16, [8, 8, 8, 8], output_map=True)

    def forward(self, x):
        xf_1_b = self.conv1_b(x)  # (1, 16, 320, 320)
        xf_2_b = self.conv2_b(x)  # (1, 16, 320, 320)
        unet1 = torch.add(xf_1_b, xf_2_b)  # (1, 16, 320, 320)
        unet1 = self.soft_boundary(unet1)
        # unet1=self.relu(unet1)
        # boundary_soft = torch.softmax(unet1, 1)
        return unet1


def crop(data1, data2, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    _, _, h2, w2 = data2.size()
    assert (h2 <= h1 and w2 <= w1)
    data = data1[:, :, crop_h:crop_h + h2, crop_w:crop_w + w2]
    return data


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class RBblock(nn.Module):
    def __init__(self, c_in):
        super(RBblock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, 64, 1, stride=1)
        self.conv2 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 1, stride=1)
        self.conv4 = nn.Conv2d(c_in, 32, 1, stride=1)

    def forward(self, x):
        a1 = self.conv1(x)
        a2 = self.conv2(a1)
        a3 = self.conv3(a2)
        a4 = self.conv4(x)
        return a4 + a3



class lastblock(nn.Module):
    def __init__(self, c_in, rate=4):
        super(lastblock,self).__init__()
        # self.rb_riddnet = RB_riddnet(c_in)
        self.rb = macnnet.ECAAttention(kernel_size=3)

        # self.att=macnnet.Attention(c_in, c_in, LayerNorm_type='WithBias')
        # self.rate = rate
        #
        # self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # dilation = self.rate*1 if self.rate >= 1 else 1
        # self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        # self.relu1 = nn.ReLU(inplace=True)
        # dilation = self.rate*2 if self.rate >= 1 else 1
        # self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        # self.relu2 = nn.ReLU(inplace=True)
        # dilation = self.rate*3 if self.rate >= 1 else 1
        # self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        # self.relu3 = nn.ReLU(inplace=True)
        # self._initialize_weights()

    def forward(self, x):
        # x=self.att(x)+x
        # x=self.rb_riddnet(x)
        x = self.rb(x)
        o = self.relu(self.conv(x))
        # o1 = self.relu1(self.conv1(o))
        # o2 = self.relu2(self.conv2(o))
        # o3 = self.relu3(self.conv3(o))
        # out = o + o1 + o2 + o3
        out = o
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class otherblock(nn.Module):
    def __init__(self, c_in, rate=4):
        super(otherblock, self).__init__()
        self.att = macnnet.CBAM(c_in)
        # self.conv = nn.Conv2d(c_in, 21, 3, stride=1, padding=1)
        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(32, 21, 3, stride=1, padding=1)




    def forward(self, x):
        # x=self.att(x)+x
        # x=self.rb_riddnet(x)
        # x = self.att(x)
        x=self.att(x)
        # print(x.shape)
        o = self.relu(self.conv(x))


        out = self.conv1(o)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
class blockw(nn.Module):
    def __init__(self, c_in, rate=4):
        super(blockw, self).__init__()
        # self.rb_riddnet = RB_riddnet(c_in)
        # self.rb = macnnet.CBAM(c_in)

        self.rb=macnnet.SimAM()
        # self.att=macnnet.Attention(c_in, c_in, LayerNorm_type='WithBias')
        # self.rate = rate
        #
        # self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = LeakyReLULayer(12,12)


        # dilation = self.rate*1 if self.rate >= 1 else 1
        # self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        # self.relu1 = nn.ReLU(inplace=True)
        # dilation = self.rate*2 if self.rate >= 1 else 1
        # self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        # self.relu2 = nn.ReLU(inplace=True)
        # dilation = self.rate*3 if self.rate >= 1 else 1
        # self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        # self.relu3 = nn.ReLU(inplace=True)
        # self._initialize_weights()

    def forward(self, x):
        # x=self.att(x)+x
        # x=self.rb_riddnet(x)
        x = self.rb(x)
        o = self.relu(self.conv(x))
        # o1 = self.relu1(self.conv1(o))
        # o2 = self.relu2(self.conv2(o))
        # o3 = self.relu3(self.conv3(o))
        # out = o + o1 + o2 + o3
        out = o
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()



class one_up(nn.Module):
    def __init__(self,cin,cout):
        super(one_up,self).__init__()
        self.upsample_lsat = nn.ConvTranspose2d(cin, cin, 4, stride=2, bias=False)
        self.conv_last_1= nn.Conv2d(cin, cout, 3, stride=1, padding=1)
        self.relu_last_1 = nn.ReLU(inplace=True)
        self.conv_last_2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1)
        self.relu_last_2= nn.ReLU(inplace=True)



    def forward(self,x):
        o=self.upsample_lsat(x)
        o=self.relu_last_1(self.conv_last_1(o))
        out=self.relu_last_2(self.conv_last_2(o))

        return out

class last_up(nn.Module):
    def __init__(self):
        super(last_up,self).__init__()
        self.up1= one_up(512, 256)
        self.up2 = one_up(256, 128)
        self.up3 = one_up(128, 64)
        self.MSB = lastblock(64, 2)
        self.conv = nn.Conv2d(32, 1, 3, stride=1, padding=1)

        # self.upsample_lsat_1 = nn.ConvTranspose2d(512, 512, 4, stride=2, bias=False)
        # self.conv_last_1_1 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        # self.relu_last_1_1 = nn.ReLU(inplace=True)
        # self.conv_last_1_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        # self.relu_last_1_2 = nn.ReLU(inplace=True)
        #
        # self.upsample_lsat_2 = nn.ConvTranspose2d(256, 256, 4, stride=2, bias=False)
        # self.conv_last_2_1 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        # self.relu_last_2_1 = nn.ReLU(inplace=True)
        # self.conv_last_1_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.relu_last_2_2 = nn.ReLU(inplace=True)
        #
        # self.upsample_lsat_3 = nn.ConvTranspose2d(128, 128, 4, stride=2, bias=False)
        # self.conv_last_3_1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        # self.relu_last_3_1 = nn.ReLU(inplace=True)
        # self.conv_last_3_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.relu_last_3_2 = nn.ReLU(inplace=True)
        #
        # self.upsample_lsat_4 = nn.ConvTranspose2d(64, 64, 4, stride=2, bias=False)
        # self.conv_last_4_1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        # self.relu_last_4_1 = nn.ReLU(inplace=True)
        # self.conv_last_4_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.relu_last_4_2 = nn.ReLU(inplace=True)



    def forward(self,x):

        x1=self.up1(x)

        x2=self.up2(x1)

        x3=self.up3(x2)




        out=self.MSB(x3)
        out = self.conv(out)

        return out










class daf(nn.Module):
    def __init__(self, pretrain=None, logger=None, rate=4):
        super(daf, self).__init__()
        # self.rb_riddnet = RB_riddnet()
        self.pretrain = pretrain
        t = 1
        # self.att=macnnet.Attention(3,3, LayerNorm_type='WithBias')
        self.features = vgg16_c.VGG16_C(pretrain, logger)
        self.blockw1_1 = blockw(64, rate)
        self.otherblock1_1 = otherblock(64, rate)

        self.blockw1_2 = blockw(64, rate)
        self.otherblock1_2 = otherblock(64, rate)

        self.conv1_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv1_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)

        self.score_dsn1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn1_1 = nn.Conv2d(21, 1, 1, stride=1)
        self.blockw2_1 = blockw(128, rate)
        self.blockw2_2 = blockw(128, rate)

        self.otherblock2_1 = otherblock(128, rate)
        self.otherblock2_2 = otherblock(128, rate)

        self.conv2_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv2_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn2 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn2_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.blockw3_1 = blockw(256, rate)
        self.blockw3_2 = blockw(256, rate)
        self.blockw3_3 = blockw(256, rate)

        self.otherblock3_1 = otherblock(256, rate)
        self.otherblock3_2 = otherblock(256, rate)
        self.otherblock3_3 = otherblock(256, rate)



        self.conv3_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv3_3_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn3 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn3_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.blockw4_1 = blockw(512, rate)
        self.blockw4_2 = blockw(512, rate)
        self.blockw4_3 = blockw(512, rate)

        self.otherblock4_1 = otherblock(512, rate)
        self.otherblock4_2 = otherblock(512, rate)
        self.otherblock4_3 = otherblock(512, rate)



        self.conv4_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv4_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv4_3_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn4 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn4_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.blockw5_1 = blockw(512, rate)
        self.blockw5_2 = blockw(512, rate)
        self.blockw5_3 = blockw(512, rate)
        self.otherblock5_1 = otherblock(512, rate)
        self.otherblock5_2 = otherblock(512, rate)
        self.otherblock5_3 = otherblock(512, rate)


        self.conv5_1_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv5_2_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.conv5_3_down = nn.Conv2d(32 * t, 21, (1, 1), stride=1)
        self.score_dsn5 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn5_1 = nn.Conv2d(21, 1, (1, 1), stride=1)

        self.upsample_2 = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
        self.upsample_4 = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.upsample_8 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upsample_8_5 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)

        self.upsample_2_ = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
        self.upsample_4_ = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.upsample_8_ = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upsample_8_5_ = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        #
        # self.lstm1=macnnet.ConvLSTMCell()
        # self.lstm2 = macnnet.ConvLSTMCell()
        # self.lstm3 = macnnet.ConvLSTMCell()
        # self.lstm4 = macnnet.ConvLSTMCell()
        # self.lstm5 = macnnet.ConvLSTMCell()

        # self.last_up=last_up()
        self.sed1=macnnet.sed(channels=1)
        self.sed2=macnnet.sed(channels=1)
        self.sed3=macnnet.sed(channels=1)
        self.sed4=macnnet.sed(channels=1)
        self.sed5=macnnet.sed(channels=1)

        #
        # self.upsample_lsat_2 = nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False)
        #
        #
        # self.upsample_lsat_3 = nn.ConvTranspose2d(128, 64, 4, stride=2, bias=False)
        #
        #
        #
        # self.upsample_lsat_3 = nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False)

        # self.att11=macnnet.Attention(21, 21, LayerNorm_type='WithBias')
        # self.att21=macnnet.Attention(21, 21, LayerNorm_type='WithBias')
        # self.att31=macnnet.Attention(21, 21, LayerNorm_type='WithBias')
        # self.att41=macnnet.Attention(21, 21, LayerNorm_type='WithBias')
        # self.att51=macnnet.Attention(21, 21, LayerNorm_type='WithBias')

        # self.CGA1_1=CGAnet.CGAFusion(64)
        # self.CGA1_2 = CGAnet.CGAFusion(64)
        # self.CGA2_1 = CGAnet.CGAFusion(128)
        # self.CGA2_2 = CGAnet.CGAFusion(128)
        # self.CGA3_1 = CGAnet.CGAFusion(256)
        # self.CGA3_2 = CGAnet.CGAFusion(256)
        # self.CGA3_3 = CGAnet.CGAFusion(256)
        # self.CGA4_1 = CGAnet.CGAFusion(512)
        # self.CGA4_2 = CGAnet.CGAFusion(512)
        # self.CGA4_3 = CGAnet.CGAFusion(512)
        # self.CGA5_1 = CGAnet.CGAFusion(512)
        # self.CGA5_2 = CGAnet.CGAFusion(512)
        # self.CGA5_3 = CGAnet.CGAFusion(512)

        # self.fuse = nn.Conv2d(10, 1, 1, stride=1)l
        # self.fuse_1 = nn.Conv2d(10, 5, 1, stride=1)
        self.fuse = nn.Conv2d(5, 1, 1, stride=1)
        # self.fuse1 = nn.Conv2d(3, 1, 1, stride=1)
        # self.fuse2 = nn.Conv2d(3, 1, 1, stride=1)
        # self.fuse3 = nn.Conv2d(3, 1, 1, stride=1)
        # self.fuse4= nn.Conv2d(3, 1, 1, stride=1)

        # # wht
        self._initialize_weights(logger)

    def forward(self, x):
        # atten_score=self.rb_riddnet(x)
        features = self.features(x)
        #     1,c,h,w
        # for i in range(len(features)):
        #     print(f"features[{i}]size:  {features[i].shape}")
        sum1 = self.conv1_1_down(self.blockw1_1(features[0])) + \
               self.conv1_2_down(self.blockw1_2(features[1]))
        # msb(ciin,32)  down (32,21)
        sum11=self.otherblock1_1(features[0])+self.otherblock1_2(features[1])


        s1 = self.score_dsn1((sum1))
        s11 = self.score_dsn1_1(sum11)

        # print(s1.data.shape, s11.data.shape)
        sum2 = self.conv2_1_down(self.blockw2_1(features[2])) + \
               self.conv2_2_down(self.blockw2_2(features[3]))

        sum21=self.otherblock2_1(features[2])+self.otherblock2_2(features[3])
        s2 = self.score_dsn2((sum2))

        s21 = self.score_dsn2_1(sum21)
        s2 = self.upsample_2(s2)
        s21 = self.upsample_2_(s21)
        # print(s2.data.shape, s21.data.shape)
        s2 = crop(s2, x, 1, 1)
        s21 = crop(s21, x, 1, 1)

        sum3 = self.conv3_1_down(self.blockw3_1(features[4])) + \
               self.conv3_2_down(self.blockw3_2(features[5]))+self.conv3_3_down(self.blockw3_3(features[6]))

        sum31 =self.otherblock3_1(features[4])+self.otherblock3_2(features[5])+self.otherblock3_3(features[6])

        s3 = self.score_dsn3((sum3))
        s3 = self.upsample_4(s3)
        # print(s3.data.shape)
        s3 = crop(s3, x, 2, 2)
        s31 = self.score_dsn3_1(sum31)
        # print(s31.shape)
        s31 = self.upsample_4_(s31)
        # print(s31.data.shape)
        s31 = crop(s31, x, 2, 2)

        sum4 = self.conv4_1_down(self.blockw4_1(features[7])) + \
               self.conv4_2_down(self.blockw4_2(features[8]))+ self.conv4_3_down(self.blockw4_3(features[9]))

        sum41 = self.otherblock4_1(features[7])+self.otherblock4_2(features[8])+self.otherblock4_3(features[9])
        s4 = self.score_dsn4((sum4))
        s4 = self.upsample_8(s4)
        # print(s4.data.shape)
        s4 = crop(s4, x, 4, 4)
        s41 = self.score_dsn4_1(sum41)
        s41 = self.upsample_8_(s41)
        # print(s41.data.shape)
        s41 = crop(s41, x, 4, 4)
        sum5 = self.conv5_1_down(self.blockw5_1(features[10])) + \
               self.conv5_2_down(self.blockw5_2(features[11]))+self.conv5_3_down(self.blockw5_3(features[12]))

        sum51 =  self.otherblock5_1(features[10])+self.otherblock5_2(features[11])+self.otherblock5_3(features[12])

        s5 = self.score_dsn5((sum5))
        s5 = self.upsample_8_5(s5)
        # print(s5.data.shape)
        s5 = crop(s5, x, 0, 0)
        s51 = self.score_dsn5_1(sum51)
        s51 = self.upsample_8_5_(s51)
        # print(s51.data.shape)
        s51 = crop(s51, x, 0, 0)

        # print(s1.shape, s11.shape, s2.shape, s21.shape,s3.shape,s31.shape,s4.shape,s41.shape,s5.shape,s51.shape)
        # print(sum1.shape,sum2.shape,sum3.shape,sum4.shape,sum5.shape)

        s11 = torch.sigmoid(s11)
        s21 = torch.sigmoid(s21)
        s31 = torch.sigmoid(s31)
        s41 = torch.sigmoid(s41)
        s51 = torch.sigmoid(s51)

        # s1 = torch.sigmoid(s1)
        # s2 = torch.sigmoid(s2)
        # s3 = torch.sigmoid(s3)
        # s4 = torch.sigmoid(s4)
        # s5 = torch.sigmoid(s5)


        # p1 = (s1 * s11 )
        # p2 = (s2 * s21)
        # p3 = (s3 * s31)
        # p4 = (s4 * s41)
        # p5 = (s5 * s51)

        # p1 = (s1 * s11 + s1)
        # p2 = (s2 * s21 + s2)
        # p3 = (s3 * s31) + s3
        # p4 = (s4 * s41) + s4
        # p5 = (s5 * s51) + s5
        #
        # up_pic=self.last_up(features[-1])
        # up_pic = crop(up_pic, x, 0, 0)

        p1=self.sed1(s1,s11)
        p2=self.sed2(s2,s21)
        p3=self.sed3(s3,s31)
        p4=self.sed4(s4,s41)
        p5=self.sed5(s5,s51)


        # p1,_=self.lstm1(p1,s1,s11)
        # p2,_=self.lstm2(p2,s2,s21)
        #
        # p3,_=self.lstm3(p3,s3,s31)
        #
        # p4,_=self.lstm4(p4,s4,s41)
        #
        # p5,_=self.lstm5(p5,s5,s51)


        # p1 = p1 * (1.0 + atten_score[:, 1, :, :].unsqueeze(1))
        # p2 = p2 * (1.0 + atten_score[:, 2, :, :].unsqueeze(1))
        # p3= p3 * (1.0 + atten_score[:, 3, :, :].unsqueeze(1))
        # p4 = p4 * (1.0 + atten_score[:, 4, :, :].unsqueeze(1))
        # p5 = p5 * (1.0 + atten_score[:, 4, :, :].unsqueeze(1))

        # p2_1 = s2 + o1
        # p3_1 = s3 + o2 + o1
        # p4_1 = s4 + o3 + o2 + o1
        # p5_1 = s5 + s4 + s3 + s2 + s1
        # p1_2 = s11 + o21 + o31 + o41 + o51
        # p2_2 = s21 + o31 + o41 + o51
        # p3_2 = s31 + o41 + o51
        # p4_2 = s41 + o51
        # p5_2 = s51

        # fuse = self.fuse(torch.cat([p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2], 1))
        #

        # a=self.fuse1(torch.cat([p1,p2,up_pic], 1))
        # b = self.fuse2(torch.cat([p2, p3,up_pic], 1))
        # c = self.fuse3(torch.cat([p3, p4,up_pic], 1))
        # d= self.fuse4(torch.cat([p4, p5,up_pic], 1))
        # fuse = self.fuse(torch.cat([p1, p2, p3, p4, p5,up_pic], 1))
        fuse = self.fuse(torch.cat([p1, p2, p3, p4, p5], 1))
        # fuse = self.fuse(torch.cat([s1, s2, s3, s4, s5], 1))

        # fuse = self.fuse(torch.cat([a, b, c, d, p5,up_pic], 1))
        # fuse = self.fuse_1(torch.cat([p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2], 1))
        # fuse = self.fuse_1(torch.cat([s1, s2, s3, s4, s5, s51, s41, s31, s21, s11], 1))
        # fuse=self.fuse_2(fuse)
        # return [s1, s2, s3, s4, s5, s51, s41, s31, s21, s11,fuse]
        # return [p1, p2, p3, p4, p5,fuse
        # ]
        # ]


        # return [s1, s2, s3, s4, s5,fuse]

        return [p1, p2, p3, p4, p5,fuse]
        # return [a, b, c, d,p5,up_pic,fuse]


        # return [p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2, fuse]

    def _initialize_weights(self, logger=None):
        for name, param in self.state_dict().items():
            if self.pretrain and 'features' in name:
                continue
            # elif 'down' in name:
            #     param.zero_()

            elif 'upsample' in name:
                if logger:
                    logger.info('init upsamle layer %s ' % name)
                # print(name)
                # print(len(name.split('.')[0]))
                if len(name.split('.'))<=2:
                    # print(name)
                    k = int(name.split('.')[0].split('_')[1])
                    param.copy_(get_upsampling_weight(1, 1, k * 2))



            elif 'fuse' in name:
                if logger:
                    logger.info('init params_autho %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    nn.init.constant_(param, 0.080)
            else:
                if logger:
                    logger.info('init params_autho %s ' % name)
                if 'bias' in name:
                    param.zero_()
                if "weight" in name:
                    param.normal_(0, 0.01)
        # print self.conv1_1_down.weight

def bspline_wavelet(x, scale):
    return (1 / 6) * F.relu(scale*x)\
    - (8 / 6) * F.relu(scale*x - (1 / 2))\
    + (23 / 6) * F.relu(scale*x - (1))\
    - (16 / 3) * F.relu(scale*x - (3 / 2))\
    + (23 / 6) * F.relu(scale*x - (2))\
    - (8 / 6) * F.relu(scale*x - (5 / 2))\
    +(1 / 6) * F.relu(scale*x - (3))

class BSplineWavelet(nn.Module):
    def __init__(self, scale=torch.as_tensor(1)):
        super().__init__()
        self.scale = torch.as_tensor(scale)

    def forward(self, x):
        output = bspline_wavelet(x, self.scale)
        return output
class LeakyReLULayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,negative_slope: float = 0.01):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.in_features = in_features
        self.out_features = out_features
    def forward(self, x):
        return self.leaky_relu(x)


