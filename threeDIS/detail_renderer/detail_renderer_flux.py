import inspect
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.attention_processor import Attention

class DetailRendererFLUX:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    counter = 0

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxAttnProcessor2_0_NPU requires PyTorch 2.0 and torch NPU, to use it, please upgrade PyTorch to 2.0 and install torch NPU"
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,  # (B, 512+((H//16)*(W//16))), 3072)
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        pos_embed=None,
        prepare_img_ids_method=None,
        instance_text_index_lst=None,
        seq_len=None,
        position_mask_list=[],
        I2I_control_steps = 4,
        I2T_control_steps = 20,
        T2I_control_steps = 20,
        T2T_control_steps = 20,
        num_inference_steps = 20,
        latent_H = None,
        latent_W = None,
    ) -> torch.FloatTensor:
        steps = num_inference_steps
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)  # (B, 512+((H//16)*(W//16))), 3072)
        key = attn.to_k(hidden_states)  # (B, 512+((H//16)*(W//16))), 3072)
        value = attn.to_v(hidden_states)  # (B, 512+((H//16)*(W//16))), 3072)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)


        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        

        def change_box_to_mask(box, H, W):
            import cv2
            c1, r1, c2, r2 = box
            # H, W = 32, 32
            c1 = int(math.floor((W * c1)))
            c2 = int(math.ceil((W * c2)))
            r1 = int(math.floor((H * r1)))
            r2 = int(math.ceil((H * r2)))
            _atten_mask = torch.zeros(H, W)
            _atten_mask[r1: r2, c1: c2] = 1
            _atten_mask = _atten_mask.reshape(H * W)
            indices_of_ones = _atten_mask.nonzero(as_tuple=True)[0]
            return indices_of_ones
        


        HW = query.shape[2] - seq_len

        H = latent_H
        W = latent_W


        # 构建atten_mask需要
        # (1) instance_text_idxs指示实例对应的text tokens的idx
        # (2) instance_bounding_boxes指示实例对应的位置
        # (3) EOT的位置

        atten_mask = torch.zeros(seq_len+HW, seq_len+HW, device=query.device)
        instance_num = len(position_mask_list)

        import torch.nn.functional as F

        assert seq_len % (instance_num + 1) == 0
        global_seq_len = seq_len // (instance_num + 1)
        


        # 限制
        for i in range(instance_num):
            instance_text_idxs = instance_text_index_lst[i].to(query.device)
            instance_img_idxs = position_mask_list[i].reshape(H * W).nonzero(as_tuple=True)[0].to(query.device)

            # 图像-->文本的mask控制
            if DetailRendererFLUX.counter % (steps * (19 + 38)) <= I2T_control_steps * (19 + 38):
                if DetailRendererFLUX.counter % (steps * (19 + 38)) <= I2I_control_steps * (19 + 38):
                    atten_mask[seq_len + instance_img_idxs, : seq_len] = 1
                else:
                    atten_mask[seq_len + instance_img_idxs, global_seq_len: seq_len] = 1

            # 图像-->图像的mask控制
            if DetailRendererFLUX.counter % (steps * (19 + 38)) <= I2I_control_steps * (19 + 38):
                atten_mask[seq_len + instance_img_idxs, seq_len:] = 1
            
            # 文本-->图像的mask控制
            if DetailRendererFLUX.counter % (steps * (19 + 38)) <= T2I_control_steps * (19 + 38):
                atten_mask[instance_text_idxs[:, None], seq_len:] = 1

            # 文本-->文本的mask控制
            if DetailRendererFLUX.counter % (steps * (19 + 38)) <= T2T_control_steps * (19 + 38):
                atten_mask[instance_text_idxs[:, None], :seq_len] = 1


        # 开启
        for i in range(instance_num):
            instance_text_idxs = instance_text_index_lst[i].to(query.device)
            instance_img_idxs = position_mask_list[i].reshape(H * W).nonzero(as_tuple=True)[0].to(query.device)

            # 图像-->文本的mask控制
            if DetailRendererFLUX.counter % (steps * (19 + 38)) <= I2T_control_steps * (19 + 38):
                atten_mask[(seq_len + instance_img_idxs)[:, None], instance_text_idxs] = 0

            # 图像-->图像的mask控制
            if DetailRendererFLUX.counter % (steps * (19 + 38)) <= I2I_control_steps * (19 + 38):
                atten_mask[(seq_len + instance_img_idxs)[:, None], seq_len + instance_img_idxs] = 0
            
            # 文本-->图像的mask控制
            if DetailRendererFLUX.counter % (steps * (19 + 38)) <= T2I_control_steps * (19 + 38):
                atten_mask[instance_text_idxs[:, None], seq_len + instance_img_idxs] = 0

            # 文本-->文本的mask控制
            if DetailRendererFLUX.counter % (steps * (19 + 38)) <= T2T_control_steps * (19 + 38):
                atten_mask[instance_text_idxs[:, None], instance_text_idxs] = 0

        atten_mask = 1 - atten_mask
        atten_mask = atten_mask.bool()

        DetailRendererFLUX.counter += 1

        

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=atten_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states