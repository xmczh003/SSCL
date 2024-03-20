#coding=utf-8
import os
import copy
import random
from collections import OrderedDict
from typing import List, Dict, Tuple, Callable, Optional, Union

import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Function
from nest import register, modules, Context

class SpatialCGNL(nn.Module):
    """Spatial CGNL block with dot production kernel for image classfication.
    """
    def __init__(self, inplanes, planes, use_scale=False, groups=None):
        self.use_scale = use_scale
        self.groups = groups
        self.inplanes = inplanes

        super(SpatialCGNL, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        if self.use_scale:
            print("=> WARN: SpatialCGNL block uses 'SCALE'", \
                   'yellow')
        if self.groups:
            print("=> WARN: SpatialCGNL block uses '{}' groups".format(self.groups), \
                   'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The linear kernel (dot production).

        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c*h*w)**0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)
        return x

    def forward(self, x, y):

        residual = x

        t = self.t(x)
        p = self.p(y)
        g = self.g(y)

        b, c, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x

class MELayer(nn.Module):
    def __init__(self, channel, reduction=16, nparts=1):
        super(MELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.nparts = nparts
        parts = list()
        for part in range(self.nparts):
            parts.append(nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
            ))
        self.parts = nn.Sequential(*parts)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)

        meouts = list()
        for i in range(self.nparts):
            meouts.append(x * self.parts[i](y).view(b, c, 1, 1))

        return x * self.parts[i](y).view(b, c, 1, 1)

def makeGaussian(size, fwhm = 3, center=None):

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

class KernelGenerator(nn.Module):
    def __init__(self, size, offset=None):
        super(KernelGenerator, self).__init__()
        
        self.size = self._pair(size) #[1]变为[1,1]，[1,1]不变
        xx, yy = np.meshgrid(np.arange(0, size), np.arange(0, size))

        if offset is None:
            offset_x = offset_y = size // 2
        else:
            offset_x, offset_y = self._pair(offset) #offset=0.
        self.factor = torch.from_numpy(-(np.power(xx - offset_x, 2) + np.power(yy - offset_y, 2)) / 2).float() #以左上角为起点计算其他点的分母

    @staticmethod
    def _pair(x):
        return (x, x) if isinstance(x, int) else x

    def forward(self, theta):
        pow2 = torch.pow(theta * self.size[0], 2)
        kernel = 1.0 / (2 * np.pi * pow2) * torch.exp(self.factor.to(theta.device) / pow2)
        return kernel / kernel.max()
def kernel_generate(theta, size, offset=None):
    return KernelGenerator(size, offset)(theta)


def _mean_filter(input):
    batch_size, num_channels, h, w = input.size()
    threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1, 1)


class PeakStimulation(Function):

    @staticmethod
    def forward(ctx, input, return_aggregation, win_size, peak_filter):

        ctx.num_flags = 4

        assert win_size % 2 == 1, 'Window size for peak finding must be odd.'
        offset = (win_size - 1) // 2 # （3-1）//2 = 1
        padding = torch.nn.ConstantPad2d(offset, float('-inf'))
        padded_maps = padding(input)
        batch_size, num_channels, h, w = padded_maps.size()
        element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
        element_map = element_map.to(input.device)
        _, indices  = F.max_pool2d(
            padded_maps,
            kernel_size = win_size,
            stride = 1,
            return_indices = True)
        peak_map = (indices == element_map)

        if peak_filter:
            mask = input >= peak_filter(input)
            peak_map = (peak_map & mask)
        peak_list = torch.nonzero(peak_map)

        ctx.mark_non_differentiable(peak_list)

        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
                peak_map.view(batch_size, num_channels, -1).sum(2)

        else:
            return peak_list



    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1)/ \
        (peak_map.view(batch_size, num_channels, -1).sum(2).view(batch_size, num_channels, 1, 1) + 1e-6)
        return (grad_input,) + (None,) * ctx.num_flags


def peak_stimulation(input, return_aggregation=True, win_size=3, peak_filter=None):

    return PeakStimulation.apply(input, return_aggregation, win_size, peak_filter)


class ScaleLayer(nn.Module):

   def __init__(self, init_value=1e-3):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, input):
       return input * self.scale


class S3N(nn.Module):

    def __init__(self, base_model, num_classes, task_input_size, base_ratio, radius, radius_inv):
        super(S3N, self).__init__()
        
        self.grid_size = 31 
        self.padding_size = 30 
        self.global_size = self.grid_size + 2*self.padding_size #91
        self.input_size_net = task_input_size #448
        gaussian_weights = torch.FloatTensor(makeGaussian(2*self.padding_size+1, fwhm = 13)) #61*61,以中心为中心的高斯核
        self.base_ratio = base_ratio  #0.09
        self.radius = ScaleLayer(radius)
        self.radius_inv = ScaleLayer(radius_inv)

        self.filter = nn.Conv2d(1, 1, kernel_size=(2*self.padding_size+1,2*self.padding_size+1),bias=False)
        self.filter.weight[0].data[:,:,:] = gaussian_weights

        self.P_basis = torch.zeros(2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size)
        for k in range(2):
            for i in range(self.global_size):
                for j in range(self.global_size):
                    self.P_basis[k,i,j] = k*(i-self.padding_size)/(self.grid_size-1.0)+(1.0-k)*(j-self.padding_size)/(self.grid_size-1.0)

        self.features = base_model.features[-1]
        self.features_raw = base_model.features[0:-2]
        self.features_coarse = base_model.features[-2]
        self.num_features = base_model.num_features

        self.raw_classifier = nn.Linear(2048, num_classes)
        self.sampler_buffer = nn.Sequential(nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            )
        self.sampler_classifier = nn.Linear(2048, num_classes)
        self.sampler_classifier2_1 = nn.Linear(2048, 2048)
        self.sampler_classifier2_2 = nn.Linear(2048, 256)

        self.sampler_classifier0_1 = nn.Linear(2048, 2048)
        self.sampler_classifier0_2 = nn.Linear(2048, 256)

        self.sampler_buffer1 = nn.Sequential(nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            )

        self.sampler_classifier1 = nn.Linear(2048, num_classes)
        self.sampler_classifier1_1 = nn.Linear(2048, 2048)
        self.sampler_classifier1_2 = nn.Linear(2048, 256)

        self.con_classifier = nn.Linear(int(self.num_features*3), num_classes)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.map_origin = nn.Conv2d(2048, num_classes, 1, 1, 0)


    def create_grid(self, x):
        P = torch.autograd.Variable(torch.zeros(1,2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size).cuda(),requires_grad=False)
        P[0,:,:,:] = self.P_basis
        P = P.expand(x.size(0),2,self.grid_size+2*self.padding_size, self.grid_size+2*self.padding_size)

        x_cat = torch.cat((x,x),1)
        p_filter = self.filter(x)
        x_mul = torch.mul(P,x_cat).view(-1,1,self.global_size,self.global_size)
        all_filter = self.filter(x_mul).view(-1,2,self.grid_size,self.grid_size)

        x_filter = all_filter[:,0,:,:].contiguous().view(-1,1,self.grid_size,self.grid_size)
        y_filter = all_filter[:,1,:,:].contiguous().view(-1,1,self.grid_size,self.grid_size)

        x_filter = x_filter/p_filter
        y_filter = y_filter/p_filter

        xgrids = x_filter*2-1
        ygrids = y_filter*2-1
        xgrids = torch.clamp(xgrids,min=-1,max=1)

        xgrids = xgrids.view(-1,1,self.grid_size,self.grid_size)
        ygrids = ygrids.view(-1,1,self.grid_size,self.grid_size)

        grid = torch.cat((xgrids,ygrids),1)

        grid = F.interpolate(grid, size=(self.input_size_net,self.input_size_net), mode='bilinear', align_corners=True)

        grid = torch.transpose(grid,1,2)
        grid = torch.transpose(grid,2,3)

        return grid

    def generate_map(self, input_x, class_response_maps, p):

        N, C, H, W = class_response_maps.size()

        score_pred, sort_number = torch.sort(F.softmax(F.adaptive_avg_pool2d(class_response_maps, 1), dim=1), dim=1, descending=True)
        gate_score = (score_pred[:, 0:5]*torch.log(score_pred[:, 0:5])).sum(1) #取熵 4*1*1 取五个是否合适？
        
        xs = []
        xs_inv = []

        for idx_i in range(N):
            if gate_score[idx_i] > -0.2:
                decide_map = class_response_maps[idx_i, sort_number[idx_i, 0],:,:]
            else:
                decide_map = class_response_maps[idx_i, sort_number[idx_i, 0:5],:,:].mean(0)
            min_value, max_value = decide_map.min(), decide_map.max()
            decide_map = (decide_map-min_value)/(max_value-min_value)

            peak_list, aggregation = peak_stimulation(decide_map, win_size=3, peak_filter=_mean_filter)
            
            decide_map = decide_map.squeeze(0).squeeze(0) #31*31 去掉多余的维度
            
            score = [decide_map[item[2], item[3]] for item in peak_list] #返回paek上的值 作为peak的score

            x = [item[3] for item in peak_list]
            y = [item[2] for item in peak_list]

            if score == []:
                temp = torch.zeros(1, 1, self.grid_size,self.grid_size).cuda() #1*1*31*31
                temp += self.base_ratio
                xs.append(temp)
                #xs_soft.append(temp)
                continue

            peak_num = torch.arange(len(score))

            temp = self.base_ratio
            temp_w = self.base_ratio

            if p == 0:
                for i in peak_num:
                    temp += score[i] * kernel_generate(self.radius(torch.sqrt(score[i])), H, (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0).cuda()

                    temp_w += 1/score[i] * \
                    kernel_generate(self.radius_inv(torch.sqrt(score[i])), H, (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0).cuda()
                    #temp = temp_w

            elif p == 1:
                for i in peak_num:
                    rd = random.uniform(0, 1)
                    if score[i] > rd:
                        temp += score[i] * kernel_generate(self.radius(torch.sqrt(score[i])), H, (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0).cuda()
                    else:
                        temp_w += 1/score[i] * \
                        kernel_generate(self.radius_inv(torch.sqrt(score[i])), H, (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0).cuda()
            elif p == 2:
                index = score.index(max(score))
                temp += score[index] * kernel_generate(self.radius(score[index]), H, (x[index].item(), y[index].item())).unsqueeze(0).unsqueeze(0).cuda()
                
                index = score.index(min(score))
                temp_w += 1/score[index] * \
                kernel_generate(self.radius_inv(torch.sqrt(score[index])), H, (x[index].item(), y[index].item())).unsqueeze(0).unsqueeze(0).cuda()

            if type(temp) == float:
                temp += torch.zeros(1, 1, self.grid_size,self.grid_size).cuda()
            xs.append(temp)
            
            if type(temp_w) == float:
                temp_w += torch.zeros(1, 1, self.grid_size,self.grid_size).cuda()
            xs_inv.append(temp_w)

        xs = torch.cat(xs, 0)
        xs_hm = nn.ReplicationPad2d(self.padding_size)(xs)
        grid = self.create_grid(xs_hm).to(input_x.device)
        x_sampled_zoom = F.grid_sample(input_x, grid)

        xs_inv = torch.cat(xs_inv, 0)
        xs_hm_inv = nn.ReplicationPad2d(self.padding_size)(xs_inv)
        grid_inv = self.create_grid(xs_hm_inv).to(input_x.device)
        x_sampled_inv = F.grid_sample(input_x, grid_inv)

        return x_sampled_zoom, x_sampled_inv

    def multi_loss(self, feature1, feature2):
        b, _, _, _ = feature1.size()
        feature1 = torch.mean(feature1, dim=1).reshape(b,-1)
        feature2 = torch.mean(feature2, dim=1).reshape(b,-1)
        similarity_matrix = torch.matmul(feature1, feature2.transpose(0,1))
        similarity_matrix = 1 + torch.exp(similarity_matrix/50.0)
        similarity = torch.diag(similarity_matrix)
        multi_loss = torch.mean(torch.log(similarity))
        return multi_loss

    def forward(self, input_x, p):



        self.map_origin.weight.data.copy_(self.raw_classifier.weight.data.unsqueeze(-1).unsqueeze(-1))
        self.map_origin.bias.data.copy_(self.raw_classifier.bias.data)
        feature_raw = self.features_raw(input_x)
        feature_coarse = self.features_coarse(feature_raw)
        feature = self.features(feature_coarse)
        agg_origin = self.raw_classifier(self.avg(feature).view(-1, 2048))

        with torch.no_grad():
            class_response_maps = F.interpolate(self.map_origin(feature), size=self.grid_size, mode='bilinear',align_corners=True)

        x_sampled_zoom, x_sampled_inv = self.generate_map(input_x, class_response_maps, p)

        feature_d = self.sampler_buffer(self.features(self.features_coarse(self.features_raw(x_sampled_zoom))))
        feature_c = self.sampler_buffer(self.features(self.features_coarse(self.features_raw(x_sampled_inv))))

        agg_sampler = self.sampler_classifier(self.avg(feature_d).view(-1, 2048))
        feature_D = self.sampler_classifier0_1(self.avg(feature_d).view(-1, 2048))
        feature_D = self.sampler_classifier0_2(feature_D)
        agg_sampler1 = self.sampler_classifier1(self.avg(feature_c).view(-1, 2048))
        feature_C = self.sampler_classifier1_1(self.avg(feature_c).view(-1, 2048))
        feature_C = self.sampler_classifier1_2(feature_C)

        feature_O = self.sampler_classifier2_1(self.avg(feature).view(-1, 2048))
        feature_O = self.sampler_classifier2_2(feature_O)

        aggregation = self.con_classifier(torch.cat([self.avg(feature).view(-1, 2048), self.avg(feature_d).view(-1, 2048), self.avg(feature_c).view(-1, 2048)], 1))




        return aggregation, agg_origin, agg_sampler, agg_sampler1, agg_origin, agg_sampler, feature_D, feature_C, feature_O

@register
def s3n(
    mode: str ='resnet50',
    num_classes: int = 200, 
    task_input_size: int = 448, 
    base_ratio: float = 0.09, 
    radius: float = 0.08, 
    radius_inv: float = 0.2) -> nn.Module:
    """ Selective sparse sampling.
    """

    classify_network = modules.ft_resnet(mode=mode, fc_or_fcn = 'fc',num_classes=num_classes)
    model = S3N(classify_network, num_classes, task_input_size, base_ratio, radius, radius_inv)

    return model


@register
def three_stage(
    ctx: Context, 
    train_ctx: Context) -> None:
    """Three stage.
    """

    if train_ctx.is_train:
        p = 0 if train_ctx.epoch_idx <= 20 else 1
    else:
        p = 1 if train_ctx.epoch_idx <= 20 else 2
    p = 1
    train_ctx.output = train_ctx.model(train_ctx.input, p)

    raise train_ctx.Skip
