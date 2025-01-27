from threeDIS.archs import RefinedShader
import torch
import numpy as np
import torch.nn as nn
from diffusers.models.attention_processor import Attention
import torch.nn.functional as F
from threeDIS.utils import get_sup_mask



class DetailRendererSD1(nn.Module):
    def __init__(self, config, place_in_unet):
        super().__init__()
        # self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.fuser = RefinedShader()        
        self.embedding = {}

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            prompt_nums=[],
            bboxes=[],
            depth=None,
            ith=None,
            embeds_pooler=None,
            timestep=None,
            height=512,
            width=512,
            ControlSteps=-1,
            masks=None,
            use_sa_preserve=False,
            sa_preserve=False,
            refined_alpha=10.0,
            depth_img_token=None,
            gamma_scale=1.0,
            sam_masks=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        assert(batch_size == 2, "We currently only implement sampling with batch_size=1, \
               and we will implement sampling with batch_size=N as soon as possible.")
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        
        instance_num = len(bboxes[0])

        is_vanilla_cross = (ith > ControlSteps)
        if instance_num == 0:
            is_vanilla_cross = True

        is_cross = encoder_hidden_states is not None
        
        ori_hidden_states = hidden_states.clone()

        # Only Need Negative Prompt and Global Prompt.
        if is_cross and is_vanilla_cross:
            encoder_hidden_states = encoder_hidden_states[:2, ...]
        
        if is_cross and not is_vanilla_cross:
            hidden_states_uncond = hidden_states[[0], ...]
            sot = encoder_hidden_states[0, 0, ...]
            hidden_states_cond = hidden_states[[1], ...].repeat(instance_num + 1, 1, 1)
            hidden_states = torch.cat([hidden_states_uncond, hidden_states_cond])

        # QKV Operation of Vanilla Self-Attention or Cross-Attention
        query = attn.to_q(hidden_states)
        
        if (
            not is_cross
            and use_sa_preserve
            and timestep.item() in self.embedding
            and self.place_in_unet == "up"
        ):
            hidden_states = torch.cat((hidden_states, torch.from_numpy(self.embedding[timestep.item()]).to(hidden_states.device)), dim=1)

        if not is_cross and sa_preserve and self.place_in_unet == "up":
            self.embedding[timestep.item()] = ori_hidden_states.cpu().numpy()
            print(self.embedding[timestep.item()].shape)

        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)  # 48 4096 77
        # self.attnstore(attention_probs, is_cross, self.place_in_unet)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        ###### Self-Attention Results ######
        if not is_cross:  
            return hidden_states

        ###### Vanilla Cross-Attention Results ######
        if is_vanilla_cross:
            return hidden_states
        
        ###### Detail Renderer ######
        assert (not is_vanilla_cross)
        # hidden_states: torch.Size([1+1+instance_num, HW, C]), the first 1 is the uncond ca output, the second 1 is the global ca output.
        hidden_states_uncond = hidden_states[[0], ...]  # torch.Size([1, HW, C])
        cond_ca_output = hidden_states[1: , ...].unsqueeze(0)  # torch.Size([1, 1+instance_num, 5, 64, 1280])
        guidance_masks = []
        in_box = []
        # Construct Instance Guidance Mask
        if masks is None and sam_masks is None:
            for bbox in bboxes[0]:  
                guidance_mask = np.zeros((height, width))
                w_min = int(width * bbox[0])
                w_max = int(width * bbox[2])
                h_min = int(height * bbox[1])
                h_max = int(height * bbox[3])
                guidance_mask[h_min: h_max, w_min: w_max] = 1.0
                guidance_masks.append(guidance_mask[None, ...])
                in_box.append([bbox[0], bbox[2], bbox[1], bbox[3]])
        elif sam_masks is not None:
            guidance_masks = [o for o in sam_masks]
        else:
            guidance_masks = [o[None, ...] for o in masks[0]]
            for bbox in bboxes[0]:
                in_box.append([bbox[0], bbox[2], bbox[1], bbox[3]])
        
        guidance_masks = np.concatenate(guidance_masks, axis=0)
        guidance_masks = guidance_masks[None, ...]
        guidance_masks = torch.from_numpy(guidance_masks).float().to(cond_ca_output.device)
        guidance_masks = F.interpolate(guidance_masks, (height//8, width//8), mode='bilinear')  # (1, instance_num, H, W)
        guidance_masks = guidance_masks.to(hidden_states.device)
            
        other_info = {}
        other_info['height'] = height
        other_info['width'] = width
        other_info['refined_alpha'] = refined_alpha

        hidden_states_cond, fuser_info = self.fuser(cond_ca_output,
                                        guidance_masks,
                                        other_info=other_info,
                                        return_fuser_info=True,)
        hidden_states_cond = hidden_states_cond.squeeze(1)

        hidden_states = torch.cat([hidden_states_uncond, hidden_states_cond])
        return hidden_states