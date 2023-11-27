from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ..utils import _topk, _tranpose_and_gather_feat
from utils.ddd_utils import get_pc_hm
from utils.pointcloud import generate_pc_hm

import torch
from torch import nn

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        self.opt = opt
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        
        self.num_stacks = num_stacks
        self.heads = heads
        self.secondary_heads = opt.secondary_heads
        # NOTE：opt.secondary_heads = ['velocity', 'nuscenes_att', 'dep_sec', 'rot_sec']
        
        last_channels = {head: last_channel for head in heads} # primary heads
        for head in self.secondary_heads:#secondary heads
          last_channels[head] = last_channel + len(opt.pc_feat_lvl)# 在原本基础上增加雷达特征通道
          # NOTE：opt.pc_feat_lvl = ['pc_dep','pc_vx','pc_vz']，如果有雷达相关，那通道数增加len()个数量
        
        for head in self.heads:
          classes = self.heads[head]
          head_conv = head_convs[head]
          if len(head_conv) > 0:
            out = nn.Conv2d(head_conv[-1], classes, 
                  kernel_size=1, stride=1, padding=0, bias=True)
            conv = nn.Conv2d(last_channels[head], head_conv[0],
                              kernel_size=head_kernel, 
                              padding=head_kernel // 2, bias=True)
            convs = [conv]
            for k in range(1, len(head_conv)):
                convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                              kernel_size=1, bias=True))
            '''
            convs的结构是[conv1(64, 256, 1x1), conv2... out(256, , classes, 1x1)]
            下面将每个conv部分后面加上激活函数，并最后放上out(1x1的卷积)
            '''
            if len(convs) == 1:
              fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
            elif len(convs) == 2:
              fc = nn.Sequential(
                convs[0], nn.ReLU(inplace=True), 
                convs[1], nn.ReLU(inplace=True), out)
            elif len(convs) == 3:
              fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), 
                  convs[2], nn.ReLU(inplace=True), out)
            elif len(convs) == 4:
              fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), 
                  convs[2], nn.ReLU(inplace=True), 
                  convs[3], nn.ReLU(inplace=True), out)
            if 'hm' in head:
              fc[-1].bias.data.fill_(opt.prior_bias)
            else:
              fill_fc_weights(fc)
          else:
            fc = nn.Conv2d(last_channels[head], classes, 
                kernel_size=1, stride=1, padding=0, bias=True)
            if 'hm' in head:
              fc.bias.data.fill_(opt.prior_bias)
            else:
              fill_fc_weights(fc)
          
          # 类实例的每个属性进行赋值时，都会首先调用__setattr__()方法，并在__setattr__()方法中将属性名和属性值添加到类实例的__dict__属性中
          # 可以理解为对每个head的实现方法（函数）进行了注册，方便后面调用
          # 比如这样子调用：self.__getattr__(head)(sec_feats)，其中sec_feats是输入（input）
          self.__setattr__(head, fc)


    def img2feats(self, x):
      raise NotImplementedError
    

    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError


    '''
    names: forward
    description: 最终模型推理调用函数
    param {*} self
    param {*} x
    param {*} pc_hm
    param {*} pc_dep
    param {*} calib
    return {*}
    '''
    def forward(self, x, pc_hm=None, pc_dep=None, calib=None):
      ## extract features from image
      feats = self.img2feats(x)
      out = []
      
      for s in range(self.num_stacks): # self.num_stacks = 1
        z = {}

        ## Run the first stage heads
        # 图像阶段检测头
        for head in self.heads: 
          if head not in self.secondary_heads: # 执行一阶段检测头
            z[head] = self.__getattr__(head)(feats[s])

        # 引入毫米波点云head，获取对应的结果
        if self.opt.pointcloud: # 雷达点云存在时生成radar heatmap和second head
          ## get pointcloud heatmap
          # 推理模式下，首先需生成hm 
          # 训练模式下，已经提前预处理好[trainer.py: LINE: 124]）
          if not self.training:
            if self.opt.disable_frustum:
              pc_hm = pc_dep
              if self.opt.normalize_depth:
                pc_hm[self.opt.pc_feat_channels['pc_dep']] /= self.opt.max_pc_dist
            else:
              # 截锥关联并生成hm
              pc_hm = generate_pc_hm(z, pc_dep, calib, self.opt)
              
          ind = self.opt.pc_feat_channels['pc_dep']
          z['pc_hm'] = pc_hm[:,ind,:,:].unsqueeze(1)

          ## Run the second stage heads  
          sec_feats = [feats[s], pc_hm] # 二阶段检测头【数据特征加上了毫米波点云】
          sec_feats = torch.cat(sec_feats, 1)
          for head in self.secondary_heads:  # 执行二阶段检测头
            z[head] = self.__getattr__(head)(sec_feats)
        
        out.append(z)

      return out

