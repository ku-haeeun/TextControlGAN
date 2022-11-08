# -*- coding: utf-8 -*-
"""
Created on Thu May  6 22:40:03 2021

@author: MinYoung
"""


# import tflib as lib

# import numpy as np
# import tensorflow as tf

# def Batchnorm(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True, labels=None, n_labels=None):
#     """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
#     if axes != [0,2,3]:
#         raise Exception('unsupported')
#     mean, var = tf.nn.moments(inputs, axes, keepdims=True)
#     shape = mean.get_shape().as_list() # shape is [1,n,1,1]
#     '''
#     offset_m = lib.param(name+'.offset', np.zeros([n_labels,shape[1]], dtype='float32'))
#     scale_m = lib.param(name+'.scale', np.ones([n_labels,shape[1]], dtype='float32'))
#     offset = tf.nn.embedding_lookup(offset_m, labels)
#     scale = tf.nn.embedding_lookup(scale_m, labels)
#     '''
#     offset = lib.ops.linear.Linear(name+'.offset', n_labels, shape[1], tf.one_hot(labels, n_labels), s_norm=True)
#     scale = lib.ops.linear.Linear(name+'.scale', n_labels, shape[1], tf.one_hot(labels, n_labels), s_norm=True) + 1.
#     result = tf.nn.batch_normalization(inputs, mean, var, offset[:,:,None,None], scale[:,:,None,None], 1e-5)
#     return result


import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


import torch.nn.utils.spectral_norm as spectral_norm

def gradient_penalty(critic, real, fake, labels, k= 1, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, labels)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - k) ** 2)
    return gradient_penalty

def _gradient_penalty(critic, real_data, generated_data, labels, device="cpu"):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data).to(device)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True).to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = critic(interpolated, labels)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return 1.0 * ((gradients_norm - 1) ** 2).mean()

def weight_penalty(critic, k= 1, device="cpu"):
    weights_norms = torch.zeros(1).to(device)
    for param in critic.parameters():
        weights_norms += (torch.sum(F.relu(param.view(param.shape[0], -1).norm(2, dim=1) - k) ** 2))
    return weights_norms

from torchvision import transforms

def get_mobilenet_score(mobilenet, x, epoch):

    preprocess = transforms.Resize(224)(x)

    prediction = F.softmax(mobilenet(preprocess), dim=1)

    p_y = torch.mean(prediction, dim= 0)
    e = prediction * torch.log(prediction / p_y)
    kl = torch.mean(torch.sum(e, dim= 1), dim= 0)
    
    return torch.exp(kl)


class _NormBase_CBN(Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps',
                     'num_features', 'affine']
    num_features: int
    num_labels: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        num_features: int,
        num_labels: int,
        eps: float = 1e-4,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ) -> None:
        super(_NormBase_CBN, self).__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = spectral_norm(nn.Linear(self.num_labels, self.num_features, bias=False))
            torch.nn.init.normal_(self.weight.weight, 0., 0.005)
            self.b = spectral_norm(nn.Linear(self.num_labels, self.num_features, bias=False))
            torch.nn.init.normal_(self.b.weight, 0., 0.005)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[operator]
            self.running_var.fill_(1)  # type: ignore[operator]
            self.num_batches_tracked.zero_()  # type: ignore[operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        '''
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.b)
        '''

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_NormBase_CBN, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class _BatchNorm_CBN(_NormBase_CBN):

    def __init__(self, num_features, num_labels, eps=1e-4, momentum=None, affine=True,
                 track_running_stats=False):
        super(_BatchNorm_CBN, self).__init__(
            num_features, num_labels, eps, momentum, affine, track_running_stats)

    def forward(self, input: Tensor, labels: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.affine:
            self.weight_CBN = self.weight(labels) + 1.
            self.b_CBN = self.b(labels)

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        mean = input.mean([0, 2, 3])  # along channel axis
        var = input.var([0, 2, 3])

        current_mean = mean.view([1, self.num_features, 1, 1]).expand_as(input)
        current_var = var.view([1, self.num_features, 1, 1]).expand_as(input)
        current_weight = self.weight_CBN.view([-1, self.num_features, 1, 1]).expand_as(input)
        current_bias = self.b_CBN.view([-1, self.num_features, 1, 1]).expand_as(input)

        return current_weight * (input - current_mean) / (current_var + self.eps).sqrt() + current_bias



class CondBatchNorm2d(_BatchNorm_CBN):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))



def one_hot_embedding(labels, num_classes, device='cpu'):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    labels = labels.type(torch.long)
    y = torch.eye(num_classes).to(device)
    return y[labels]

def init_weights(m):
    if type(m) == nn.Conv2d:
        #torch.nn.init.xavier_uniform_(m.weight, gain=2**0.5)
        #torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.orthogonal_(m.weight)

def init_weights_sc(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.orthogonal_(m.weight)
        

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm= True, spec_norm= False, resample= None, num_cls=25):
        super(ResidualBlock, self).__init__()
                
        # (Normalization) - Activation - Convolution -(Normalization) - Activation - Convolution
        layers = []

        self.batch_norm = batch_norm

        self.conv1, self.conv2, self.shortcut = self._make_conv_layer(in_channels, out_channels, kernel_size, stride, padding, batch_norm, spec_norm, resample)
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        self.shortcut.apply(init_weights_sc)
        '''
        if batch_norm:
            layers.append(CondBatchNorm2d(in_channels))
        layers.append(nn.ReLU(inplace= True))
        layers.append(conv1)
        if batch_norm:
            layers.append(CondBatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace= True))
        layers.append(conv2)

        self.features = nn.Sequential( *layers )
        '''
        self.nonlinearF = nn.ReLU(inplace= True)
        if batch_norm:
            self.cbn1 = CondBatchNorm2d(in_channels, num_cls)
            self.cbn2 = CondBatchNorm2d(out_channels, num_cls)
        
    def forward(self, x, labels):
        if self.shortcut == None:
            if self.batch_norm:
                out = self.cbn1(x, labels)
                out = self.nonlinearF(out)
                out = self.conv1(out)
                out = self.cbn2(out, labels)
                out = self.nonlinearF(out)
                out = self.conv2(out)
                return out + x
            else:
                out = self.nonlinearF(x)
                out = self.conv1(out)
                out = self.nonlinearF(out)
                out = self.conv2(out)
                return out + x
        else:
            if self.batch_norm:
                out = self.cbn1(x, labels)
                out = self.nonlinearF(out)
                out = self.conv1(out)
                out = self.cbn2(out, labels)
                out = self.nonlinearF(out)
                out = self.conv2(out)
                return out + self.shortcut(x)
            else:
                out = self.nonlinearF(x)
                out = self.conv1(out)
                out = self.nonlinearF(out)
                out = self.conv2(out)
                return out + self.shortcut(x)
    
    
    # If you use batch norm, that means you don't need to use bias parameters in convolution layers
    def _make_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm, spec_norm, resample):
        
        if resample == 'up':
            
            conv1 = nn.Sequential(
                                    nn.UpsamplingNearest2d(scale_factor= 2),
                                    spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)),
                                            
                                    )
            conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False))
            
                
            kernel_size, padding, bias = 1, 0, True
            shortcut = nn.Sequential(
                                        nn.UpsamplingNearest2d(scale_factor= 2),
                                        spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias= False)),
                                                
                                        )
                
                
        elif resample == 'down':

            if spec_norm:
                
                conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias= True))
                conv2 = nn.Sequential( 
                                        spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias= True)),
                                        nn.AvgPool2d((2, 2))
                                        
                                        )
                
                kernel_size, padding, bias = 1, 0, True
                shortcut = nn.Sequential( 
                                            spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias= True)),
                                            nn.AvgPool2d((2, 2))
                                            
                                            )
                
                
            else:
                
                conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias= True)
                conv2 = nn.Sequential( 
                                        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias= True),
                                        nn.AvgPool2d((2, 2))
                                        
                                        )
                
                kernel_size, padding, bias = 1, 0, True
                shortcut = nn.Sequential( 
                                            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias= True),
                                            nn.AvgPool2d((2, 2))
                                            
                                            )   
            
                
        elif resample == None:
            
            if spec_norm:
                
                conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias= True))
                conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias= True))
                shortcut = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias= True))
                
            
            else:
            
                conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias= True)
                conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias= True)
                shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias= True)
            
        return conv1, conv2, shortcut
    

class CriticFirstBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, spec_norm= True):
        super(CriticFirstBlock, self).__init__()

        layers = []
        
        conv1, conv2, self.shortcut = self._make_conv_layer(in_channels, out_channels, kernel_size, stride, padding, spec_norm)

        conv1.apply(init_weights)
        conv2.apply(init_weights)
        self.shortcut.apply(init_weights_sc)

        layers.append(conv1)
        layers.append(nn.ReLU(inplace= True))
        layers.append(conv2)


        self.features = nn.Sequential(*layers)



    def forward(self, x):
        return self.features(x) + self.shortcut(x)
        
        
    def _make_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, spec_norm):
        
        
        if spec_norm:
            
            conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias= True))
            conv2 = nn.Sequential( 
                                    spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias= True)),
                                    nn.AvgPool2d((2, 2))
                                                    
                                    )
            kernel_size, padding = 1, 0
            shortcut = nn.Sequential(
                                    nn.AvgPool2d((2, 2)),
                                    spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias= True))

                                    )
            
        else:
            
            conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias= True)
            conv2 = nn.Sequential( 
                                    nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias= True),
                                    nn.AvgPool2d((2, 2))
                                                    
                                    )
            kernel_size, padding = 1, 0
            shortcut = nn.Sequential(
                                    nn.AvgPool2d((2, 2)),
                                    nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias= True)
                                                    
                                    )   

        return conv1, conv2, shortcut


