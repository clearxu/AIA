"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn.functional as F
import math
from detectron2.utils import comm

import open_clip

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

import torch.distributions as tdist

from torch import nn

import random
from timm.models.layers import DropPath

from mmcv.runner import BaseModule

from einops import rearrange
from functools import partial
from torch import nn, einsum

from .attmask import AttMask
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli
import sys
import math

from .ScConv import SRU


class FocusedDropoutAdapter(nn.Module):
    def __init__(self, in_channels, low=0.6, high=0.9):
        super(FocusedDropoutAdapter, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)


    def random_drop(self, mask, p):
        drop_mask = torch.rand_like(mask.float())  # 将 mask 转换为 float 类型
        drop_mask[drop_mask < p] = 0
        drop_mask[drop_mask >= p] = 1
        return mask * drop_mask.byte()  # 将结果掩码转换回 bool 类型

    def forward(self, x):

        average_channel = torch.mean(x, dim=(1), keepdim=True)

        percentile_value = 90
        threshold_value = torch.quantile(average_channel.float(), percentile_value / 100.0)
        mask = average_channel > threshold_value

        random_values = torch.rand_like(mask.float())

        mask[random_values < 0.5] = 0

        drop_mask = ~mask.bool()

        return x * drop_mask

class MonaOp_dc(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

        self.prompt = torch.nn.parameter.Parameter(torch.randn(in_features, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(in_features), requires_grad=True)


        self.relu = nn.ReLU(inplace=True)
        self.drop_area1 = FocusedDropoutAdapter(in_channels=in_features)

        self.drop_area2 = FocusedDropoutAdapter(in_channels=in_features)
        self.drop_area3 = FocusedDropoutAdapter(in_channels=in_features)

    def forward(self, x):

        b, c, h,w = x.shape

        identity = x


        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = self.drop_area1(conv1_x + conv2_x + conv3_x) / 3.0 + identity


        identity = x

        x = self.projector(x)
        x = self.relu(x)

        x = x.reshape(b, c, -1).permute(0, 2, 1)



        cos_sim = F.normalize(x, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)
        # B, N, 1
        mask = cos_sim.clamp(0, 1)
        x = x * mask
        x = x @ self.top_down_transform

        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2) + identity

        return x

class MonaOp_dd(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

        self.prompt = torch.nn.parameter.Parameter(torch.randn(in_features, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(in_features), requires_grad=True)

        self.nonlinear = F.relu
        self.scp1 = SelectiveChannelPruning(in_channels=in_features, channel_scorer_channels=in_features)
        self.scp2 = SelectiveChannelPruning(in_channels=in_features, channel_scorer_channels=in_features)
        self.scp3 = SelectiveChannelPruning(in_channels=in_features, channel_scorer_channels=in_features)


    def forward(self, x,dd):
        # h, w = hw
        b, c, h,w = x.shape
        # x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        identity = x
        # conv1_x = self.scp1(self.conv1(x))
        # conv2_x = self.scp2(self.conv2(x))
        # conv3_x = self.scp3(self.conv3(x))

        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = self.scp1(conv1_x + conv2_x + conv3_x, dd) / 3.0 + identity
        # x = conv1_x + conv2_x + conv3_x / 3.0 + identity

        identity = x
        x = self.projector(x)

        x = x.reshape(b, c, -1).permute(0, 2, 1)

        x = self.nonlinear(x)

        cos_sim = F.normalize(x, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)
        mask = cos_sim.clamp(0, 1)
        x = x * mask
        x = x @ self.top_down_transform

        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2) + identity
        # x = x.reshape(b, c, -1).permute(0, 2, 1)
        return x

class SelectiveChannelPruning(nn.Module):
    def __init__(self, in_channels, channel_scorer_channels, q_percentile=90):
        super(SelectiveChannelPruning, self).__init__()

        self.q_percentile = q_percentile


    def mask_selection(self, scores, percent, wrs_flag):
        batch_size, num_neurons = scores.shape
        drop_num = int(num_neurons * percent)
        if wrs_flag == 0:

            x_max = F.softmax(scores)

            mask_filters = 1 - Bernoulli(x_max).sample()
            # threshold = torch.sort(scores, dim=1, descending=True)[0][:, drop_num]
            # mask_filters = (scores > threshold.view(batch_size, 1)).float()
        else:
            score_max = scores.max(dim=1, keepdim=True)[0]
            score_min = scores.min(dim=1, keepdim=True)[0]
            scores = (scores - score_min) / (score_max - score_min)

            r = torch.rand_like(scores)
            key = r.pow(1. / scores)
            threshold = torch.sort(key, dim=1, descending=True)[0][:, drop_num]
            mask_filters = (key > threshold.view(batch_size, 1)).float()

        mask_filters = 1 - mask_filters
        return mask_filters

    def get_scores(self, feature, score, percent=0.01):
        right_score = feature * score.unsqueeze(-1).unsqueeze(-1)
        right_score = (right_score - right_score.min(dim=1, keepdim=True)[0]) / \
                      (right_score.max(dim=1, keepdim=True)[0] - right_score.min(dim=1, keepdim=True)[0])
        return self.mask_selection(right_score.mean([2, 3]), percent, wrs_flag=0)



    def ortho_channel(self, input):
        N, C, H, W = input.shape
        vec = input.view(N, C, H * W)
        vec = vec / (torch.sqrt(torch.sum(torch.pow(vec, 2), dim=-1, keepdim=True)) + 1e-8)
        # print(vec)
        # assert False
        P = torch.abs(torch.matmul(vec, torch.transpose(vec, 1, 2)) - torch.eye(C).to(input.device).view(1, C, C))
        # print(torch.matmul(vec, torch.transpose(vec, 1, 2)))
        # print(P)
        rank = torch.sum(P, dim=-1) / (C)
        rank = rank.view(N, C)
        ortho_scores = 1 - self.normalize_scores(rank)
        # print(rank)
        return ortho_scores

    def ortho_channel1(self, input):
        N, C, H, W = input.shape
        vec = input.view(N, C, H * W)
        vec = vec / (torch.sqrt(torch.sum(torch.pow(vec, 2), dim=-1, keepdim=True)) + 1e-8)
        # print(vec)
        # assert False
        P = torch.abs(torch.matmul(vec, torch.transpose(vec, 1, 2)) - torch.eye(C).to(input.device).view(1, C, C))
        rank = -torch.sum(P, dim=-1) / (C)
        rank = rank.view(N, C)

        ortho_scores = self.normalize_scores(rank)

        return ortho_scores

    def compute_correlation(self, x_a, x):
        N, C, H, W = x_a.shape

        # Reshape the input tensor to a 3D tensor
        vec_input = x_a.view(N, C, H * W)

        # Normalize the vectors along the last dimension
        vec_input = vec_input / (torch.sqrt(torch.sum(torch.pow(vec_input, 2), dim=-1, keepdim=True)) + 1e-8)

        # Reshape the second tensor to a 3D tensor
        vec_x = x.view(N, x.shape[1], -1)

        # Normalize the vectors along the last dimension
        vec_x = vec_x / (torch.sqrt(torch.sum(torch.pow(vec_x, 2), dim=-1, keepdim=True)) + 1e-8)

        # Calculate the correlation matrix P between input and x
        P = torch.abs(torch.matmul(vec_input, torch.transpose(vec_x, 1, 2)))

        # Calculate the rank of P along the last dimension and normalize the scores
        rank = torch.sum(P, dim=-1) / (C)
        rank = rank.view(N, C)
        correlation_scores = self.normalize_scores(rank)

        return correlation_scores

    def get_activations(self, inputs):
        return torch.mean(inputs, dim=[2, 3])

    def normalize_scores(self, scores):
        return (scores - scores.min()) / (scores.max() - scores.min())

    def forward(self, x,dd):
        b,c,h,w = x.shape
        feature = x.clone().detach()
        channel_wise_activation = x.mean([2, 3])  # Global Average Pooling


        channel_importance_scores = self.compute_correlation(x,dd)

        channel_importance_scores = F.softmax(channel_importance_scores)
        # print(x.shape,channel_importance_scores.shape)

        mask_filters = self.get_scores(feature=feature, score=channel_importance_scores)
        mask_filters = mask_filters.view(*mask_filters.shape, 1, 1)
        return mask_filters * x






@BACKBONE_REGISTRY.register()
class CLIP(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        model_name = cfg.MODEL.FC_CLIP.CLIP_MODEL_NAME
        pretrained = cfg.MODEL.FC_CLIP.CLIP_PRETRAINED_WEIGHTS
        pretrained = '/data2/wcw/.cache/huggingface/hub/models--laion--CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/snapshots/4070cadc6220ffa4df32772e528ec11e7cb73780/open_clip_pytorch_model.bin'


        # download on local rank 0 first
        if comm.get_local_rank() == 0:
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        comm.synchronize()

        self.model_name = model_name
        self.pretrained = pretrained

        self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.text_tokenizer = open_clip.get_tokenizer(model_name)

        model_name = model_name.lower()
        if 'convnext_' in model_name:
            self.model_type = 'convnext'
            if '_base' in model_name:
                self.output_channels = [128, 128, 256, 512, 1024]
            elif '_large' in model_name:
                self.output_channels = [192, 192, 384, 768, 1536]
            elif '_xxlarge' in model_name:
                self.output_channels = [384, 384, 768, 1536, 3072]

        elif 'rn' in model_name:
            self.model_type = 'resnet'
            if model_name.replace('-quickgelu', '') in ['rn50', 'rn101']:
                self.output_channels = [64, 256, 512, 1024, 2048]
            elif model_name == 'rn50x4':
                self.output_channels = [80, 320, 640, 1280, 2560]
            elif model_name == 'rn50x16':
                self.output_channels = [96, 384, 768, 1536, 3072]
            elif model_name == 'rn50x64':
                self.output_channels = [128, 512, 1024, 2048, 4096]

        self._out_feature_strides = {
            "stem": 2,
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
            "clip_embedding": -1
        }
        self._out_feature_channels = {
            "stem": self.output_channels[0],
            "res2": self.output_channels[1],
            "res3": self.output_channels[2],
            "res4": self.output_channels[3],
            "res5": self.output_channels[4],
            "clip_embedding": self.dim_latent
        }

        # self.interaction = ContextInteraction(q_dim=768,
        #                                                    k_dim=768,
        #                                                    embed_dim=768,
        #                                                    num_heads=8,
        #                                                    hidden_dim=768,
        #                                                    use_layer_scale=True)

        # self.mona = Mona(in_dim=768, factor=4)
        # self.mona = MonaOp_mask(in_features=768)
        # self.mona = MonaOp_dc(in_features=768)
        # self.gm = EfficientAtt(dim=768)
        # self.scp = SelectiveChannelPruning(in_channels=192,channel_scorer_channels=192)
        self.mona_dd = MonaOp_dd(in_features=192)

        self.mona_dc = MonaOp_dc(in_features=768)

        # self.a = nn.Parameter(torch.randn(1.))
        #
        # self.b = nn.Parameter(torch.randn(1.))

        # self.eval()
        self.freeze_everything()

    def freeze_everything(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.clip_model.transformer.get_cast_dtype()

        x = self.clip_model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def tokenize_text(self, text):
        return self.text_tokenizer(text)

    def extract_features(self, x,text_classifier,num_templates=None):
        return {
            'convnext': self.extract_features_convnext,
            'resnet': self.extract_features_resnet,
        }[self.model_type](x,text_classifier,num_templates)

    def visual_prediction_forward(self, x, masks=None):
        return {
            'convnext': self.visual_prediction_forward_convnext,
            'resnet': self.visual_prediction_forward_resnet,
        }[self.model_type](x, masks)

    def extract_features_convnext(self, x,text_classifier, num_templates):
        out = {}
        x = self.clip_model.visual.trunk.stem(x)
        out['stem'] = x.contiguous()  # os4

        id = x.clone()
        for i in range(4):
            if i == 0:
                dd = self.clip_model.visual.trunk.stages[i](x)
            id = self.clip_model.visual.trunk.stages[i](id)
            # out[f'res{i + 2}'] = id.contiguous()  # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)

        id = self.clip_model.visual.trunk.norm_pre(id)

        # id = get_classification_logits(id,text_classifier,logit_scale=1.0, num_templates=num_templates)
        out['clip_vis_dense_id'] = id.contiguous()
        for i in range(4):
            # self.scp.requires_grad_(False)
            if i == 0:
                # self.scp.requires_grad_(True)
                # if True:
                # with torch.enable_grad():
                # mask_filters = self.scp(x)
                # x = x * mask_filters + x
                x = self.clip_model.visual.trunk.stages[i](x)
                x = self.mona_dd(x, dd)
                out[f'res{i + 2}'] = x.contiguous() # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)
            elif i == 2:
                x = self.clip_model.visual.trunk.stages[i](x)
                x = self.mona_dc(x)
                # torch.Size([4, 384, 100, 160])
                # torch.Size([4, 768, 50, 80])
                # print(s.shape, x.shape)
                out[f'res{i + 2}'] = x.contiguous()   # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)

            else:
                x = self.clip_model.visual.trunk.stages[i](x)
                out[f'res{i + 2}'] = x.contiguous()  # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)

        x = self.clip_model.visual.trunk.norm_pre(x)
        out['clip_vis_dense'] = x.contiguous()
        return out

    def extract_features_resnet(self, x):
        out = {}
        x = self.clip_model.visual.act1(self.clip_model.visual.bn1(self.clip_model.visual.conv1(x)))
        x = self.clip_model.visual.act2(self.clip_model.visual.bn2(self.clip_model.visual.conv2(x)))
        x = self.clip_model.visual.act3(self.clip_model.visual.bn3(self.clip_model.visual.conv3(x)))
        out['stem'] = x.contiguous()  # os2
        x = self.clip_model.visual.avgpool(x)
        x = self.clip_model.visual.layer1(x)
        out['res2'] = x.contiguous()  # os4
        x = self.clip_model.visual.layer2(x)
        out['res3'] = x.contiguous()  # os8
        x = self.clip_model.visual.layer3(x)
        out['res4'] = x.contiguous()  # os16
        x = self.clip_model.visual.layer4(x)
        out['res5'] = x.contiguous()  # os32
        out['clip_vis_dense'] = x
        return out

    def visual_prediction_forward_convnext(self, x, masks):
        batch, num_query, channel = x.shape
        x = x.reshape(batch * num_query, channel, 1, 1)  # fake 2D input
        x = self.clip_model.visual.trunk.head(x)
        x = self.clip_model.visual.head(x)
        return x.view(batch, num_query, x.shape[-1])  # B x num_queries x 640

    def visual_prediction_forward_resnet(self, x, masks):
        batch, channel, height, width = x.shape
        if masks.shape[-2] != height or masks.shape[-1] != width:
            masks = F.inteprolate(masks, size=(height, width), mode='bilinear', align_corners=False)
        num_masks = masks.shape[1]

        positional_embedding = self.clip_model.visual.attnpool.positional_embedding.to(x.dtype)
        spatial_pos_embed = positional_embedding[1:, None, :]  # HW x 1 x C
        orig_size = int(math.sqrt(spatial_pos_embed.shape[0]))
        spatial_pos_embed = spatial_pos_embed.permute(1, 2, 0).reshape(1, channel, orig_size, orig_size)
        spatial_pos_embed = F.interpolate(spatial_pos_embed, size=(height, width), mode='bilinear',
                                          align_corners=False)  # 1 x C x H x W
        spatial_pos_embed = spatial_pos_embed.permute(2, 3, 0, 1).reshape(height * width, 1, channel)
        x = x.reshape(batch, channel, height * width).permute(2, 0, 1)  # BCHW -> (HW)BC
        key_value = x + spatial_pos_embed

        masks = masks.reshape(batch, num_masks, height * width)
        masks = (masks > 0).to(masks.dtype)
        query = x.mean(0, keepdim=True) + positional_embedding[:1, None, :]
        query = query.repeat_interleave(num_masks, dim=0)

        attn_mask = masks < 0.5
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.clip_model.visual.attnpool.num_heads, -1, -1)
        attn_mask = attn_mask.reshape(batch * self.clip_model.visual.attnpool.num_heads,
                                      query.shape[0], key_value.shape[0])

        x = F.multi_head_attention_forward(
            query=query, key=key_value, value=key_value,
            embed_dim_to_check=key_value.shape[-1],
            num_heads=self.clip_model.visual.attnpool.num_heads,
            q_proj_weight=self.clip_model.visual.attnpool.q_proj.weight,
            k_proj_weight=self.clip_model.visual.attnpool.k_proj.weight,
            v_proj_weight=self.clip_model.visual.attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.clip_model.visual.attnpool.q_proj.bias,
                                    self.clip_model.visual.attnpool.k_proj.bias,
                                    self.clip_model.visual.attnpool.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.clip_model.visual.attnpool.c_proj.weight,
            out_proj_bias=self.clip_model.visual.attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.clip_model.visual.attnpool.training,
            need_weights=False,
            attn_mask=attn_mask
        )[0].permute(1, 0, 2)  # B x N x C

        return x

    def get_text_classifier(self, text_list, device):
        self.eval()
        with torch.no_grad():
            # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
            text_tokens = self.tokenize_text(text_list)
            text_tokens = text_tokens.to(device)
            # we return un-normalized text feature.
            text_features = self.encode_text(text_tokens, normalize=False)
            return text_features

    def forward(self, x, text_classifier=None, num_templates=None):
        # self.eval()
        # with torch.no_grad():
        #     return self.extract_features(x,text_classifier)
        return self.extract_features(x, text_classifier,num_templates)

    @property
    def dim_latent(self):
        return self.clip_model.text_projection.shape[-1]

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in ["stem", "res2", "res3", "res4", "res5", "clip_embedding"]
        }

    @property
    def size_divisibility(self):
        return -1