import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.modules import MSDeformAttn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_

from .base.vit import TIMMVisionTransformer
from .comer_modules import CNN, CTIBlock, deform_inputs

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


@BACKBONE_REGISTRY.register()
class ViTCoMer(TIMMVisionTransformer, Backbone):
    def __init__(self, cfg, input_shape):

        pretrain_size = cfg.MODEL.VIT_COMER.PRETRAINED_SIZE
        num_heads = cfg.MODEL.DINOV2.NUM_HEADS
        conv_inplane = cfg.MODEL.VIT_COMER.CONV_INPLANE
        n_points = cfg.MODEL.VIT_COMER.N_POINTS
        deform_num_heads = cfg.MODEL.VIT_COMER.DEFORM_NUM_HEADS
        init_values = cfg.MODEL.VIT_COMER.INIT_VALUES
        interaction_indexes = cfg.MODEL.VIT_COMER.INTERACTION_INDEXES
        with_cffn = cfg.MODEL.VIT_COMER.WITH_CFFN
        cffn_ratio = cfg.MODEL.VIT_COMER.CFFN_RATIO
        deform_ratio = cfg.MODEL.VIT_COMER.DEFORM_RATIO
        use_CTI_toV = cfg.MODEL.VIT_COMER.USE_CTI_TOV
        use_CTI_toC = cfg.MODEL.VIT_COMER.USE_CTI_TOC
        cnn_feature_interaction = cfg.MODEL.VIT_COMER.CNN_FEATURE_INTERACTION

        add_vit_feature = True
        use_extra_CTI = True
        extra_num = 4

        super().__init__(cfg)

        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.use_CTI_toC = use_CTI_toC
        self.use_CTI_toV = use_CTI_toV
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = CNN(inplanes=conv_inplane,
                                      embed_dim=embed_dim)
        self.interactions = nn.Sequential(*[
            CTIBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                            init_values=init_values, drop_path=self.drop_path_rate,
                            norm_layer=self.norm_layer, with_cffn=with_cffn,
                            cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                            use_CTI_toV=use_CTI_toV if isinstance(use_CTI_toV, bool) else use_CTI_toV[i],
                            use_CTI_toC=use_CTI_toC if isinstance(use_CTI_toC, bool) else use_CTI_toC[i],
                            cnn_feature_interaction=cnn_feature_interaction if isinstance(cnn_feature_interaction, bool) else cnn_feature_interaction[i],
                            extra_CTI=((True if i == len(interaction_indexes) - 1 else False) and use_extra_CTI),
                            extra_num=extra_num)
            for i in range(len(interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        self._out_features = ['res2', 'res3', 'res4', 'res5']

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.embed_dim,
            "res3": self.embed_dim,
            "res4": self.embed_dim,
            "res5": self.embed_dim,
        }

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        return {
            'res2': f1,
            'res3': f2,
            'res4': f3,
            'res5': f4,
        }
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
    
    @property
    def size_divisibility(self):
        return 32