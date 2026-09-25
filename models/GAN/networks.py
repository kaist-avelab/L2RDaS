import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
# import spconv.pytorch as spconv
from functools import partial
from utils.spconv_utils import replace_feature, spconv
import matplotlib.pyplot as plt
import torch.nn.functional as F

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        # netG.cuda(gpu_ids[0])
        netG.cuda()
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        # netD.cuda(gpu_ids[0])
        netD.cuda()
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real, mask=None):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                if mask is not None:
                    mask_resized = F.interpolate(mask, size=pred.shape[2:], mode='nearest')
                    pred = pred[mask_resized == 1]
                    target_tensor = target_tensor[mask_resized == 1]

                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            if mask is not None:
                mask_resized = F.interpolate(mask, size=input[-1].shape[2:], mode='nearest')
                input[-1] = input[-1][mask_resized == 1]
                target_tensor = target_tensor[mask_resized == 1]
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y, mask=None):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            if mask is not None:
                mask_resized = F.interpolate(mask, size=x_vgg[i].shape[2:], mode='nearest')
                mask_resized = mask_resized.expand(-1, x_vgg[i].shape[1], -1, -1, -1)
                loss += self.weights[i] * self.criterion(x_vgg[i][mask_resized==1], y_vgg[i][mask_resized==1].detach())
            else:
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            # loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm3d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad3d(3), nn.Conv3d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv3d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose3d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                               norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad3d(3), nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool3d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

#----------original GlobalGenerator-------------
# class GlobalGenerator(nn.Module):
#     def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
#                  padding_type='reflect'):
#         assert(n_blocks >= 0)
#         super(GlobalGenerator, self).__init__()
#         activation = nn.ReLU(True)

#         model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
#         ### downsample
#         for i in range(n_downsampling):
#             mult = 2**i
#             model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
#                       norm_layer(ngf * mult * 2), activation]

#         ### resnet blocks
#         mult = 2**n_downsampling
#         for i in range(n_blocks):
#             model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

#         ### upsample
#         for i in range(n_downsampling):
#             mult = 2**(n_downsampling - i)
#             model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
#                        norm_layer(int(ngf * mult / 2)), activation]
#         model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
#         self.model = nn.Sequential(*model)

#     def forward(self, input):
#         return self.model(input)
#
def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    elif conv_type == 'transSparseConv':
        conv = spconv.SparseConvTranspose3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    elif conv_type == 'transconv':
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1, bias=False)
        m = nn.Sequential(
            conv,
            norm_fn(out_channels),
            nn.ReLU(),
        )
        return m
    elif conv_type == 'conv3d':
        conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding, bias=False)
        m = nn.Sequential(
            conv,
            norm_fn(out_channels),
            nn.ReLU(),
        )
        return m
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1), bias=False)
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1), bias=False)
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1), bias=False)


    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm1d,
                 padding_type='zero'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        norm_fn_3d = partial(nn.BatchNorm3d, eps=1e-3, momentum=0.01)

        # self.sparse_shape =[33, 192, 192]
        self.sparse_shape=[33,192,192]

        #model = [nn.ReflectionPad3d(3), nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_nc, ngf, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(ngf),
            nn.ReLU(),
        )
        ### downsample
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16*2, 16*2, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [33, 96, 192] -> [17 48 96]
            block(16*2, 32*2, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32*2, 32*2, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32*2, 32*2, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            #  [17 48 96] <- [9, 24, 48]
            block(32*2, 64*2, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64*2, 64*2, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64*2, 64*2, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [9, 24, 48] <- [5, 12, 24]
            block(64*2, 128*2, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(128*2, 128*2, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(128*2, 128*2, 3,  norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        # Adding this to reduce z-dimension from 5 to 4
        # [5, 12, 24] -> [4, 12, 24]
        self.reduce_z_dim = spconv.SparseSequential(
            spconv.SparseConv3d(128*2, 128*2, (2, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                bias=False, indice_key='spconv_down_z'),
            norm_fn(128*2),
            nn.ReLU(),
        )

        ## resnet blocks
        use_bias=None
        self.conv5 = spconv.SparseSequential(
            SparseBasicBlock(128*2, 128*2, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(128*2, 128*2, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(128*2, 128*2, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(128*2, 128*2, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(128*2, 128*2, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(128*2, 128*2, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(128*2, 128*2, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(128*2, 128*2, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(128*2, 128*2, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        # Adding this to reduce z-dimension from 5 to 4
        # [4, 12, 24] -> [5, 12, 24]
        self.increase_z_dim = nn.Sequential(
            nn.ConvTranspose3d(128*2, 128*2, (2, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                bias=False),
            norm_fn_3d(128*2),
            nn.ReLU(),
        )

        ## upsample
        # self.conv6 = nn.Sequential(
        #     nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     norm_fn_3d(64),
        #     nn.ReLU()
        # )
        # [5, 12, 24] -> [9, 24, 48]
        self.conv6 = nn.Sequential(
            block((128+128)*2, 64*2, 3, norm_fn=norm_fn_3d, stride=2, padding=1, conv_type='transconv'),
            block(64*2, 64*2, 3, norm_fn=norm_fn_3d, padding=1, conv_type='conv3d'),
            block(64*2, 64*2, 3, norm_fn=norm_fn_3d, padding=1, conv_type='conv3d'),
            block(64*2, 64*2, (2,1,1), norm_fn=norm_fn_3d, stride=(1,1,1), padding=(0,0,0), conv_type='conv3d')
        )
        # [9, 24, 48]-> [17, 48, 96]
        self.conv7 = nn.Sequential(
            block((64+64)*2, 32*2, 3, norm_fn=norm_fn_3d, stride=2, padding=1, conv_type='transconv'),
            block(32*2, 32*2, 3, norm_fn=norm_fn_3d, padding=1, conv_type='conv3d'),
            block(32*2, 32*2, 3, norm_fn=norm_fn_3d, padding=1, conv_type='conv3d'),
            block(32*2, 32*2, (2,1,1), norm_fn=norm_fn_3d, stride=(1,1,1), padding=(0,0,0), conv_type='conv3d')
        )
        # [17, 48, 96]-> [33, 96, 192]
        self.conv8 = nn.Sequential(
            block((32+32)*2, 16*2, 3, norm_fn=norm_fn_3d, stride=2, padding=1, conv_type='transconv'),
            block(16*2, 16*2, 3, norm_fn=norm_fn_3d, padding=1, conv_type='conv3d'),
            block(16*2, 16*2, 3, norm_fn=norm_fn_3d, padding=1, conv_type='conv3d'),
            block(16*2, 16*2, (2,1,1), norm_fn=norm_fn_3d, stride=(1,1,1), padding=(0,0,0), conv_type='conv3d')
        )

        # self.conv7 = nn.Sequential(
        #     nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     norm_fn_3d(32),
        #     nn.ReLU()
        # )

        # self.conv8 = nn.Sequential(
        #     nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     norm_fn_3d(16),
        #     nn.ReLU()
        # )

        # self.conv6 = UpConv3DBlock(in_channels=128, res_channels=64)
        # self.conv7 = UpConv3DBlock(in_channels=64, res_channels=32)
        # self.conv8 = UpConv3DBlock(in_channels=32, res_channels=16)
        # [33, 96, 192]-> [32, 96, 192]
        self.conv9 = nn.Sequential(
            block(16*2, 16*2, 3, norm_fn=norm_fn_3d, padding=1, conv_type='conv3d'),
            block(16*2, 16*2, (2,1,1), norm_fn=norm_fn_3d, stride=(1,1,1), padding=(0,0,0), conv_type='conv3d')
        )
        # [32, 96, 192]-> [32, 96, 192]
        self.conv_out = nn.Sequential(
            nn.Conv3d(ngf, output_nc, kernel_size=3, padding=1),
            nn.ReLU()
            # nn.Tanh()
        )


    def forward(self, input, dict_item):
        input = self.conv_input(input)
        #down sample
        input = self.conv1(input)
        x_conv2 = self.conv2(input)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4_ = self.reduce_z_dim(x_conv4)

        #resblock
        output = self.conv5(x_conv4_)

        #upsample
        # dense_x=x_conv5.dense()
        output=self.increase_z_dim(output.dense())
        output = self.conv6(torch.cat((output, x_conv4.dense()),1))
        output = self.conv7(torch.cat((output, x_conv3.dense() ),1))
        output = self.conv8(torch.cat((output, x_conv2.dense()),1))
        output = self.conv9(output)
        output = self.conv_out(output)

        #######------------------vis----------------------------#############
        # output_2d = output.mean(dim=2)[0, 0].cpu().detach().numpy()
        # plt.imshow(output_2d, cmap='gray')
        # plt.axis('off')
        # plt.show()
        return output


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, activation, norm_layer):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, activation, norm_layer)

    def build_conv_block(self, dim, padding_type, activation, norm_layer):
        conv_block = []

        conv_block += [
            spconv.SubMConv3d(dim, dim, kernel_size=3, padding=1, bias=False, indice_key='subm_res'),
            activation(dim),
            nn.ReLU()
        ]

        conv_block += [
            spconv.SubMConv3d(dim, dim, kernel_size=3, padding=1, bias=False, indice_key='subm_res'),
            activation(dim)
        ]

        return spconv.SparseSequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm3d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad3d(3), nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad3d(3), nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm3d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool3d(3, stride=2, padding=[1, 1,1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm3d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        norm_fn_3d = partial(nn.BatchNorm3d, eps=1e-3, momentum=0.01)

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_fn_3d(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_fn_3d(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        # Define 3D Convolution layers based on VGG19 architecture
        self.slice1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        )

        self.slice2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        )

        self.slice3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        )

        self.slice4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        )

        self.slice5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out