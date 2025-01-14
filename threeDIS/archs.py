import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from threeDIS.layers import CBAM, CrossAttention, LayoutAttention


class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)  # torch.Size([5, 30, 64])


class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        # -------------------------------------------------------------- #
        self.linears_position = nn.Sequential(
            nn.Linear(self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, boxes):

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes)  # B*1*4 --> B*1*C torch.Size([5, 1, 64])
        xyxy_embedding = self.linears_position(xyxy_embedding)  # B*1*C --> B*1*768 torch.Size([5, 1, 768])

        return xyxy_embedding


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears_position = nn.Sequential(
            nn.Linear(768 + 64, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 768),
        )

    def forward(self, boxes):

        xyxy_embedding = self.linears_position(boxes)  # B*1*C --> B*1*768 torch.Size([5, 1, 768])

        return xyxy_embedding


class RefinedShader(nn.Module):
    # https://arxiv.org/pdf/2407.02329
    def __init__(self):
        super().__init__()
    def forward(self, ca_x, guidance_mask, other_info, ca_scale=None, return_fuser_info=False):
        # ca_x: (B, instance_num+1, HW, C)
        # guidance_mask: (B, instance_num, H, W)
        # box: (instance_num, 4)
        # image_token: (B, instance_num+1, HW, C)
        full_H = other_info['height']
        full_W = other_info['width']
        B, _, HW, C = ca_x.shape
        instance_num = guidance_mask.shape[1]
        down_scale = int(math.sqrt(full_H * full_W // ca_x.shape[2]))
        H = full_H // down_scale
        W = full_W // down_scale
        guidance_mask = guidance_mask.clone()
        guidance_mask = guidance_mask * other_info.get('refined_alpha', 10.0)
        guidance_mask = F.interpolate(guidance_mask, size=(H, W), mode='bilinear')   # (B, instance_num, H, W)
        guidance_mask = torch.cat([torch.ones(B, 1, H, W).to(guidance_mask.device), guidance_mask ], dim=1)  # (B, instance_num+1, H, W)
        if ca_scale is not None:
            assert len(ca_scale) == instance_num
            for i in range(instance_num):
                ca_x[:, i+1, ...] = ca_x[:, i+1, ...] * ca_scale[i]
                
                
        guidance_mask = guidance_mask.type_as(ca_x)
            
        guidance_mask = guidance_mask.view(B, instance_num + 1, HW, 1)
        guidance_mask = guidance_mask.softmax(dim=1)
        out_MIGC = (ca_x * guidance_mask).sum(dim=1, keepdim=True)
        if return_fuser_info:
            return out_MIGC, None
        else:
            return out_MIGC
        
        
class SAC(nn.Module):
    def __init__(self, C, number_pro=30):
        super().__init__()
        self.C = C
        self.number_pro = number_pro
        self.conv2 = nn.Conv2d(C, 1, 1, 1)
        self.cbam2 = CBAM(number_pro, reduction_ratio=1)

    def forward(self, x, guidance_mask=None, sac_scale=None, other_info=None):
        '''
        :param x: (B, phase_num, HW, C)
        :param guidance_mask: (B, phase_num, H, W)
        :return:
        '''
        B, phase_num, HW, C = x.shape
        H = other_info['feature_height']
        W = other_info['feature_width']

        null_x = torch.zeros_like(x[:, [0], ...]).to(x.device)

        x = torch.cat([x, null_x], dim=1)
        phase_num += 1

        scale = x  # (B, phase_num, HW, C)
        scale = scale.view(-1, H, W, C)  # (B * phase_num, H, W, C)
        scale = scale.permute(0, 3, 1, 2)  # (B * phase_num, C, H, W)
        scale = self.conv2(scale)  # (B * phase_num, 1, H, W)
        scale = scale.view(B, phase_num, H, W)  # (B, phase_num, H, W)

        null_scale = scale[:, [-1], ...]
        scale = scale[:, :-1, ...]
        x = x[:, :-1, ...]

        pad_num = self.number_pro - phase_num + 1
        ori_phase_num = scale[:, :-1, ...].shape[1]
        phase_scale = torch.cat([scale[:, :-1, ...], null_scale.repeat(1, pad_num, 1, 1)], dim=1)
        shuffled_order = torch.randperm(phase_scale.shape[1])
        inv_shuffled_order = torch.argsort(shuffled_order)

        random_phase_scale = phase_scale[:, shuffled_order, ...]

        scale = torch.cat([random_phase_scale, scale[:, [-1], ...]], dim=1)
        # (B, number_pro, H, W)

        scale = self.cbam2(scale)  # (B, number_pro, H, W)
        scale = scale.view(B, self.number_pro, HW)[..., None]  # (B, number_pro, HW)

        random_phase_scale = scale[:, : -1, ...]
        phase_scale = random_phase_scale[:, inv_shuffled_order[:ori_phase_num], :]
        if sac_scale is not None:
            instance_num = len(sac_scale)
            for i in range(instance_num):
                phase_scale[:, i, ...] = phase_scale[:, i, ...] * sac_scale[i]


        scale = torch.cat([phase_scale, scale[:, [-1], ...]], dim=1)

        scale = scale.softmax(dim=1)  # (B, phase_num, HW, 1)
        out = (x * scale).sum(dim=1, keepdims=True)  # (B, 1, HW, C)
        return out, scale


class MIGC(nn.Module):
    def __init__(self, C, attn_type='base', context_dim=768, heads=8):
        super().__init__()
        self.ea = CrossAttention(query_dim=C, context_dim=context_dim,
                             heads=heads, dim_head=C // heads,
                             dropout=0.0)
        self.la = LayoutAttention(query_dim=C,
                                    heads=heads, dim_head=C // heads,
                                    dropout=0.0)
        self.norm = nn.LayerNorm(C)
        self.sac = SAC(C)
        self.pos_net = PositionNet(in_dim=768, out_dim=768)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, ca_x, guidance_mask, other_info, return_fuser_info=False):
        # x: (B, instance_num+1, HW, C)
        # guidance_mask: (B, instance_num, H, W)
        # box: (B, instance_num, 4)
        # image_token: (B, instance_num+1, HW, C)
        ca_scale = other_info['ca_scale'] if 'ca_scale' in other_info else None
        ea_scale = other_info['ea_scale'] if 'ea_scale' in other_info else None
        mig_ca_x = ca_x[:, [0], ...]
        full_H = other_info['height']
        full_W = other_info['width']
        B, _, HW, C = ca_x.shape
        instance_num = guidance_mask.shape[1]
        down_scale = int(math.sqrt(full_H * full_W // ca_x.shape[2]))
        H = full_H // down_scale
        W = full_W // down_scale
        other_info['feature_height'] = H
        other_info['feature_width'] = W

        guidance_mask = F.interpolate(guidance_mask, size=(H, W), mode='bilinear')   # (B, instance_num, H, W)

        supplement_mask = other_info['supplement_mask']  # (B, 1, 64, 64)
        supplement_mask = F.interpolate(supplement_mask, size=(H, W), mode='bilinear')  # (B, 1, H, W)
        image_token = other_info['image_token']
        assert image_token.shape == ca_x.shape
        context = other_info['context_pooler']
        box = other_info['box']
        # box = self.pos_net.fourier_embedder(box)
        box_token = self.pos_net(box.reshape(B * instance_num, 1, -1))
        if context.ndim == 3:
            context = torch.cat([context[1:, ...].reshape(B * instance_num, 1, -1), box_token], dim=1)
        else:
            context = torch.cat([context[:, 1:, ...].reshape(B * instance_num, 1, -1), box_token], dim=1)
        sac_scale = other_info['sac_scale'] if 'sac_scale' in other_info else None

        ea_x, ea_attn = self.ea(self.norm(image_token[:, 1:, ...].reshape(B * instance_num, HW, C)),
                                     context=context, return_attn=True)
        ea_x = ea_x.view(B, instance_num, HW, C)
        ea_x = ea_x * guidance_mask.view(B, instance_num, HW, 1)
        if ea_scale is not None:
            assert len(ea_scale) == instance_num
            for i in range(instance_num):
                ea_x[:, i, ...] = ea_x[:, i, ...] * ea_scale[i]

        ori_image_token = image_token[:, 0, ...]  # (B, HW, C)
        fusion_template = self.la(x=ori_image_token, guidance_mask=torch.cat([guidance_mask[:, :, ...], supplement_mask], dim=1))  # (B, HW, C)
        fusion_template = fusion_template.view(B, 1, HW, C)  # (B, 1, HW, C)

        shading_instances_and_template = torch.cat([ea_x, fusion_template], dim = 1)
        # ca_x[:, 0, ...] = ca_x[:, 0, ...] * supplement_mask.view(B, HW, 1)
        # guidance_mask = torch.cat([
        #     guidance_mask,
        #     torch.ones(B, 1, H, W).to(guidance_mask.device)
        #     ], dim=1)

        out_MIGC, sac_scale = self.sac(shading_instances_and_template, sac_scale=sac_scale, other_info=other_info)  # (B, 1, HW, C)
        out_MIGC = mig_ca_x + out_MIGC * torch.tanh(self.gamma) * other_info.get('gamma_scale', 1.0)
        if return_fuser_info:
            fuser_info = {}
            fuser_info['sac_scale'] = sac_scale.view(B, instance_num + 1, H, W)
            fuser_info['ea_attn'] = ea_attn.mean(dim=1).view(B, instance_num, H, W, 2)
            return out_MIGC, fuser_info
        else:
            return out_MIGC
        
