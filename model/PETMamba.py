import torch
import torch.nn as nn

from ops.utils import soft_threshold 
from tqdm import tqdm
import  numpy as np
import pdb
from .mamba import *

norm_layer = nn.LayerNorm

class PETMamba(nn.Module):
    def __init__(self, args):
        super(PETMamba, self).__init__()
        self.inner_num = args.inner_num
        self.iter_num = args.iter_num
        self.frame_num = args.frame_num
        
        self.md1 = CZSS(hidden_dim=self.frame_num * 3, patch=args.patch, drop_path=args.drop_rate, norm_layer=nn.LayerNorm, 
                                 attn_drop_rate=args.attn_drop_rate, d_state=args.d_state)
        self.mc1 = CZSS(hidden_dim=self.inner_num * 3, patch=args.patch, drop_path=args.drop_rate, norm_layer=nn.LayerNorm, 
                                 attn_drop_rate=args.attn_drop_rate, d_state=args.d_state)

        self.d2 = nn.Conv3d(self.frame_num, self.inner_num, kernel_size=3, padding=1, stride=1, bias=False)
        self.dn2 = nn.BatchNorm3d(self.inner_num)
        self.d3 = nn.Conv3d(self.inner_num, self.inner_num, kernel_size=3, padding=1, stride=1, bias=False)
        self.dn3 = nn.BatchNorm3d(self.inner_num)

        self.c2 = nn.Conv3d(self.inner_num, self.inner_num, kernel_size=3, padding=1, bias=False)
        self.cn2 = nn.BatchNorm3d(self.inner_num)
        self.c3 = nn.Conv3d(self.inner_num, self.frame_num, kernel_size=3, padding=1, bias=False)
        self.cn3 = nn.BatchNorm3d(self.frame_num)
        
        model = []
        for i in range(self.iter_num):
            model += [PETMambaBlock(args)]
        self.model = nn.Sequential(*model)

    def forward(self, x):      
        x1 = self.dn3(self.d3(self.dn2(self.d2(self.md1(x)))))

        for i in range(self.iter_num):
            x1 = self.model[i](x1, x)
        
        x1 = x1.permute(0, 2, 3, 1)

        x1 = self.cn3(self.c3(self.cn2(self.c2(self.mc1(x1)))))

        return x1

class PETMambaBlock(nn.Module):
    def __init__(self, args):
        super(PETMambaBlock, self).__init__()

        self.d2 = nn.Conv3d(args.inner_num, args.inner_num, kernel_size=3, padding=1, bias=False)
        self.dn2 = nn.BatchNorm3d(args.inner_num)

        self.d3 = nn.Conv3d(args.inner_num, args.frame_num, kernel_size=3, padding=1, bias=False)
        self.dn3 = nn.BatchNorm3d(args.frame_num)

        self.c2 = nn.Conv3d(args.frame_num, args.inner_num, kernel_size=3, padding=1, bias=False)
        self.cn2 = nn.BatchNorm3d(args.inner_num)

        self.c3 = nn.Conv3d(args.inner_num, args.inner_num, kernel_size=3, padding=1, bias=False)
        self.cn3 = nn.BatchNorm3d(args.inner_num)

        self.soft_threshold = soft_threshold
        self.lmbda = nn.Parameter(torch.zeros(1, args.inner_num, 1, 1, 1))
        nn.init.constant_(self.lmbda, 0.02)

        self.m1 = CZSS(hidden_dim=args.inner_num * 3, patch=args.patch, drop_path=args.drop_rate, norm_layer=nn.LayerNorm, 
                                 attn_drop_rate=args.attn_drop_rate, d_state=args.d_state)
        self.m2 = CZSS(hidden_dim=args.frame_num * 3, patch=args.patch, drop_path=args.drop_rate, norm_layer=nn.LayerNorm, 
                                 attn_drop_rate=args.attn_drop_rate, d_state=args.d_state)
    
    def forward(self, x, g):
        x1 = self.dn3(self.d3(self.dn2(self.d2(self.m1(x)))))

        x1 = g - x1

        x1 = self.cn3(self.c3(self.cn2(self.c2(self.m2(x1)))))

        x1 = x + x1
        x1 = self.soft_threshold(x1, self.lmbda)
        return x1






