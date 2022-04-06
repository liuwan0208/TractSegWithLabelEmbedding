
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from tractseg.libs.pytorch_utils import conv2d
from tractseg.libs.pytorch_utils import deconv2d


class UNet_Pytorch_DeepSup_Simp1(torch.nn.Module):
    def __init__(self, embed_dim, n_input_channels=3, n_classes=7, n_filt=64, batchnorm=False, dropout=False,upsample="bilinear"):
        super(UNet_Pytorch_DeepSup_Simp1, self).__init__()

        self.use_dropout = dropout
        self.in_channel = n_input_channels
        self.n_classes = n_classes


        self.embed_dim = embed_dim
        print('embed dim is', self.embed_dim)

        # self.contr_1 = conv2d(n_classes, n_classes, batchnorm=batchnorm)
        self.contr_1 = nn.Conv2d(n_classes, self.embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_6 = nn.Conv2d(self.embed_dim, n_classes, kernel_size=1, stride=1, padding=0, bias=True)


        self.contr_1_1_seg = conv2d(n_input_channels, n_filt, batchnorm=batchnorm)
        self.contr_1_2_seg = conv2d(n_filt, n_filt, batchnorm=batchnorm)
        self.pool_1_seg = nn.MaxPool2d((2, 2))

        self.contr_2_1_seg = conv2d(n_filt, n_filt * 2, batchnorm=batchnorm)
        self.contr_2_2_seg = conv2d(n_filt * 2, n_filt * 2, batchnorm=batchnorm)
        self.pool_2_seg = nn.MaxPool2d((2, 2))

        self.contr_3_1_seg = conv2d(n_filt * 2, n_filt * 4, batchnorm=batchnorm)
        self.contr_3_2_seg = conv2d(n_filt * 4, n_filt * 4, batchnorm=batchnorm)
        self.pool_3_seg = nn.MaxPool2d((2, 2))

        self.contr_4_1_seg = conv2d(n_filt * 4, n_filt * 8, batchnorm=batchnorm)
        self.contr_4_2_seg = conv2d(n_filt * 8, n_filt * 8, batchnorm=batchnorm)
        self.pool_4_seg = nn.MaxPool2d((2, 2))

        self.dropout_seg  = nn.Dropout(p=0.4)

        self.encode_1_seg  = conv2d(n_filt * 8, n_filt * 16, batchnorm=batchnorm)
        self.encode_2_seg  = conv2d(n_filt * 16, n_filt * 16, batchnorm=batchnorm)
        self.deconv_1_seg  = deconv2d(n_filt * 16, n_filt * 16, kernel_size=2, stride=2)

        self.expand_1_1_seg  = conv2d(n_filt * 8 + n_filt * 16, n_filt * 8, batchnorm=batchnorm)
        self.expand_1_2_seg  = conv2d(n_filt * 8, n_filt * 8, batchnorm=batchnorm)
        self.deconv_2_seg  = deconv2d(n_filt * 8, n_filt * 8, kernel_size=2, stride=2)

        self.expand_2_1_seg  = conv2d(n_filt * 4 + n_filt * 8, n_filt * 4, stride=1, batchnorm=batchnorm)
        self.expand_2_2_seg  = conv2d(n_filt * 4, n_filt * 4, stride=1, batchnorm=batchnorm)
        self.deconv_3_seg  = deconv2d(n_filt * 4, n_filt * 4, kernel_size=2, stride=2)


        self.expand_3_1_seg  = conv2d(n_filt * 2 + n_filt * 4, n_filt * 2, stride=1, batchnorm=batchnorm)
        self.expand_3_2_seg  = conv2d(n_filt * 2, n_filt * 2, stride=1, batchnorm=batchnorm)
        self.deconv_4_seg  = deconv2d(n_filt * 2, n_filt * 2, kernel_size=2, stride=2)

        self.expand_4_1_seg = conv2d(n_filt + n_filt * 2, n_filt, stride=1, batchnorm=batchnorm)
        self.expand_4_2_seg = conv2d(n_filt, n_filt, stride=1, batchnorm=batchnorm)
        self.conv_5_seg = nn.Conv2d(n_filt, self.embed_dim, kernel_size=1, stride=1, padding=0, bias=True)


        # Deep Supervision
        self.output_2_seg = nn.Conv2d(n_filt * 4 + n_filt * 8, self.embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.output_2_up_seg = nn.Upsample(scale_factor=2, mode=upsample)
        self.output_3_seg = nn.Conv2d(n_filt * 2 + n_filt * 4, self.embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.output_3_up_seg = nn.Upsample(scale_factor=2, mode=upsample)  # does only upscale width and height



    def forward(self, input, label):
        contr_1 = self.contr_1(label)



        contr_1_1 = self.contr_1_1_seg(input)
        contr_1_2 = self.contr_1_2_seg(contr_1_1)
        pool_1 = self.pool_1_seg(contr_1_2)

        contr_2_1 = self.contr_2_1_seg(pool_1)
        contr_2_2 = self.contr_2_2_seg(contr_2_1)
        pool_2 = self.pool_2_seg(contr_2_2)

        contr_3_1 = self.contr_3_1_seg(pool_2)
        contr_3_2 = self.contr_3_2_seg(contr_3_1)
        pool_3 = self.pool_3_seg(contr_3_2)

        contr_4_1 = self.contr_4_1_seg(pool_3)
        contr_4_2 = self.contr_4_2_seg(contr_4_1)
        pool_4 = self.pool_4_seg(contr_4_2)

        if self.use_dropout:
            pool_4 = self.dropout_seg(pool_4)

        encode_1 = self.encode_1_seg(pool_4)
        encode_2 = self.encode_2_seg(encode_1)
        deconv_1 = self.deconv_1_seg(encode_2)

        concat1 = torch.cat([deconv_1, contr_4_2], 1)
        expand_1_1 = self.expand_1_1_seg(concat1)
        expand_1_2 = self.expand_1_2_seg(expand_1_1)
        deconv_2 = self.deconv_2_seg(expand_1_2)

        concat2 = torch.cat([deconv_2, contr_3_2], 1)
        expand_2_1 = self.expand_2_1_seg(concat2)
        expand_2_2 = self.expand_2_2_seg(expand_2_1)
        deconv_3 = self.deconv_3_seg(expand_2_2)

        concat3 = torch.cat([deconv_3, contr_2_2], 1)
        expand_3_1 = self.expand_3_1_seg(concat3)
        expand_3_2 = self.expand_3_2_seg(expand_3_1)
        deconv_4 = self.deconv_4_seg(expand_3_2)

        concat4 = torch.cat([deconv_4, contr_1_2], 1)
        expand_4_1 = self.expand_4_1_seg(concat4)
        expand_4_2_seg = self.expand_4_2_seg(expand_4_1)

        conv_5_seg = self.conv_5_seg(expand_4_2_seg)


        # Deep Supervision
        output_2 = self.output_2_seg(concat2)
        output_2_up = self.output_2_up_seg(output_2)
        output_3 = output_2_up + self.output_3_seg(concat3)
        output_3_up = self.output_3_up_seg(output_3)



        # Deep Supervision
        conv_5_seg =conv_5_seg+output_3_up

        conv_6_seg = self.conv_6(conv_5_seg)
        conv_6_rec = self.conv_6(contr_1)
        return conv_6_seg, conv_6_rec, contr_1, conv_5_seg
