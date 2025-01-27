import glob
import random
import time
from typing import Tuple, Any, Callable, Dict, List, Optional, Union
# import moxing as mox
import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.pipelines import StableDiffusionControlNetPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
import inspect
import os
import math
import torch.nn as nn
import torch.nn.functional as F
# from utils import load_utils
import argparse
import yaml
import cv2
import math
from scipy.ndimage import uniform_filter, gaussian_filter
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from threeDIS.utils import construct_supplement_mask_from_layout, get_sup_mask

logger = logging.get_logger(__name__)

class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down": [], "mid": [], "up": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if is_cross:
            if attn.shape[1] in self.attn_res:
                self.step_store[place_in_unet].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def maps(self, block_type: str):
        return self.attention_store[block_type]

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res=[64*64, 32*32, 16*16, 8*8]):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res

import torch.fft as fft


def Fourier_filter(x, scale=0.5, fft_d=8):
    dtype = x.dtype
    x = x.type(torch.float32)
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).fill_(scale).cuda()
    crow, ccol = H // 2, W // 2
    thresh, thresw = int(math.ceil(H * (1 / fft_d))), int(math.ceil(W * (1 / fft_d)))
    mask[..., crow - thresw : crow + thresw, ccol - thresh : ccol + thresh] = 1.0
    x_freq = x_freq * mask
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    x_filtered = x_filtered.type(dtype)
    return x_filtered




class SD1RenderingPipeline(StableDiffusionPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,        
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel] = None,
            image_encoder: CLIPVisionModelWithProjection = None,
            requires_safety_checker: bool = True,
    ):
        # Get the parameter signature of the parent class constructor
        parent_init_signature = inspect.signature(super().__init__)
        parent_init_params = parent_init_signature.parameters
        
        if controlnet is not None:
            self.register_modules(controlnet=controlnet)
        else:
            self.controlnet = None
            
        # Dynamically build a parameter dictionary based on the parameters of the parent class constructor
        init_kwargs = {
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "unet": unet,
            "scheduler": scheduler,
            "safety_checker": safety_checker,
            "feature_extractor": feature_extractor,
            "requires_safety_checker": requires_safety_checker,
        }
        if 'image_encoder' in parent_init_params.items():
            init_kwargs['image_encoder'] = image_encoder
        super().__init__(**init_kwargs)
        self.use_controlnet = controlnet is not None
        self.instance_set = set()
        self.embedding = {}
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

    def _encode_prompt(
            self,
            prompts,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompts is not None and isinstance(prompts, str):
            batch_size = 1
        elif prompts is not None and isinstance(prompts, list):
            batch_size = len(prompts)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds_none_flag = (prompt_embeds is None)
        prompt_embeds_list = []
        embeds_pooler_list = []
        for prompt in prompts:
            if prompt_embeds_none_flag:
                # textual inversion: procecss multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

                text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = self.tokenizer(
                    prompt, padding="longest", return_tensors="pt"
                ).input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[
                    -1
                ] and not torch.equal(text_input_ids, untruncated_ids):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )

                if (
                        hasattr(self.text_encoder.config, "use_attention_mask")
                        and self.text_encoder.config.use_attention_mask
                ):
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device),
                    attention_mask=attention_mask,
                )
                embeds_pooler = prompt_embeds.pooler_output
                prompt_embeds = prompt_embeds[0]

            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            embeds_pooler = embeds_pooler.to(dtype=self.text_encoder.dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            embeds_pooler = embeds_pooler.repeat(1, num_images_per_prompt)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )
            embeds_pooler = embeds_pooler.view(
                bs_embed * num_images_per_prompt, -1
            )
            prompt_embeds_list.append(prompt_embeds)
            embeds_pooler_list.append(embeds_pooler)
        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        embeds_pooler = torch.cat(embeds_pooler_list, dim=0)
        # negative_prompt_embeds: (prompt_nums[0]+prompt_nums[1]+...prompt_nums[n], token_num, token_channel), <class 'torch.Tensor'>

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                negative_prompt = "worst quality, low quality, bad anatomy"
            uncond_tokens = [negative_prompt] * batch_size

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                    hasattr(self.text_encoder.config, "use_attention_mask")
                    and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            # negative_prompt_embeds: (len(prompt_nums), token_num, token_channel), <class 'torch.Tensor'>

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            final_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return final_prompt_embeds, prompt_embeds, embeds_pooler[:, None, :]

    def check_inputs(
            self,
            prompt,
            token_indices,
            bboxes,
            height,
            width,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            controlnet_conditioning_scale=1.0,
            control_guidance_start=0.0,
            control_guidance_end=1.0,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
                callback_steps is not None
                and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
                not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if token_indices is not None:
            if isinstance(token_indices, list):
                if isinstance(token_indices[0], list):
                    if isinstance(token_indices[0][0], list):
                        token_indices_batch_size = len(token_indices)
                    elif isinstance(token_indices[0][0], int):
                        token_indices_batch_size = 1
                    else:
                        raise TypeError(
                            "`token_indices` must be a list of lists of integers or a list of integers."
                        )
                else:
                    raise TypeError(
                        "`token_indices` must be a list of lists of integers or a list of integers."
                    )
            else:
                raise TypeError(
                    "`token_indices` must be a list of lists of integers or a list of integers."
                )

        if bboxes is not None:
            if isinstance(bboxes, list):
                if isinstance(bboxes[0], list):
                    if (
                            isinstance(bboxes[0][0], list)
                            and len(bboxes[0][0]) == 4
                            and all(isinstance(x, float) for x in bboxes[0][0])
                    ):
                        bboxes_batch_size = len(bboxes)
                    elif (
                            isinstance(bboxes[0], list)
                            and len(bboxes[0]) == 4
                            and all(isinstance(x, float) for x in bboxes[0])
                    ):
                        bboxes_batch_size = 1
                    else:
                        print(isinstance(bboxes[0], list), len(bboxes[0]))
                        raise TypeError(
                            "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                        )
                else:
                    print(isinstance(bboxes[0], list), len(bboxes[0]))
                    raise TypeError(
                        "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                    )
            else:
                print(isinstance(bboxes[0], list), len(bboxes[0]))
                raise TypeError(
                    "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                )

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if token_indices_batch_size != prompt_batch_size:
            raise ValueError(
                f"token indices batch size must be same as prompt batch size. token indices batch size: {token_indices_batch_size}, prompt batch size: {prompt_batch_size}"
            )

        if bboxes_batch_size != prompt_batch_size:
            raise ValueError(
                f"bbox batch size must be same as prompt batch size. bbox batch size: {bboxes_batch_size}, prompt batch size: {prompt_batch_size}"
            )

        if self.use_controlnet:
            if isinstance(self.controlnet, MultiControlNetModel):
                if isinstance(prompt, list):
                    logger.warning(
                            f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                            " prompts. The conditionings will be fixed across the prompts."
                        )

            # Check `image`
            is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
                self.controlnet, torch._dynamo.eval_frame.OptimizedModule
            )
            if (
                isinstance(self.controlnet, ControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, ControlNetModel)
            ):
                self.check_image(image, prompt, prompt_embeds)
            elif (
                isinstance(self.controlnet, MultiControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
            ):
                if not isinstance(image, list):
                    raise TypeError("For multiple controlnets: `image` must be type `list`")

                # When `image` is a nested list:
                # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
                elif any(isinstance(i, list) for i in image):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
                elif len(image) != len(self.controlnet.nets):
                    raise ValueError(
                        f"For multiple controlnets: `image` must have the same length as the number of controlnets, but got {len(image)} images and {len(self.controlnet.nets)} ControlNets."
                    )

                for image_ in image:
                    self.check_image(image_, prompt, prompt_embeds)
            else:
                assert False

            # Check `controlnet_conditioning_scale`
            if (
                isinstance(self.controlnet, ControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, ControlNetModel)
            ):
                if not isinstance(controlnet_conditioning_scale, float):
                    raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
            elif (
                isinstance(self.controlnet, MultiControlNetModel)
                or is_compiled
                and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
            ):
                if isinstance(controlnet_conditioning_scale, list):
                    if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                        raise ValueError("A single batch of multiple conditionings are supported at the moment.")
                elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                    self.controlnet.nets
                ):
                    raise ValueError(
                        "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                        " the same length as the number of controlnets"
                    )
            else:
                assert False

            if not isinstance(control_guidance_start, (tuple, list)):
                control_guidance_start = [control_guidance_start]

            if not isinstance(control_guidance_end, (tuple, list)):
                control_guidance_end = [control_guidance_end]

            if len(control_guidance_start) != len(control_guidance_end):
                raise ValueError(
                    f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
                )

            if isinstance(self.controlnet, MultiControlNetModel):
                if len(control_guidance_start) != len(self.controlnet.nets):
                    raise ValueError(
                        f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}."
                    )

            for start, end in zip(control_guidance_start, control_guidance_end):
                if start >= end:
                    raise ValueError(
                        f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                    )
                if start < 0.0:
                    raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
                if end > 1.0:
                    raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")


    def check_image(self, image, prompt, prompt_embeds):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_np = isinstance(image, np.ndarray)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
        image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

        if (
            not image_is_pil
            and not image_is_tensor
            and not image_is_np
            and not image_is_pil_list
            and not image_is_tensor_list
            and not image_is_np_list
        ):
            raise TypeError(
                f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
            )

        if image_is_pil:
            image_batch_size = 1
        else:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        
        # print(image.shape)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def get_indices(self, prompt: str) -> Dict[str, int]:
        """Utility function to list the indices of the tokens you wish to alte"""
        ids = self.tokenizer(prompt).input_ids
        indices = {
            i: tok
            for tok, i in zip(
                self.tokenizer.convert_ids_to_tokens(ids), range(len(ids))
            )
        }
        return indices

    @staticmethod
    def draw_box(pil_img: Image, bboxes: List[List[float]]) -> Image:
        """Utility function to draw bbox on the image"""
        width, height = pil_img.size
        draw = ImageDraw.Draw(pil_img)

        for obj_box in bboxes:
            x_min, y_min, x_max, y_max = (
                obj_box[0] * width,
                obj_box[1] * height,
                obj_box[2] * width,
                obj_box[3] * height,
            )
            draw.rectangle(
                [int(x_min), int(y_min), int(x_max), int(y_max)],
                outline="red",
                width=4,
            )

        return pil_img


    @staticmethod
    def draw_box_desc(pil_img: Image, bboxes: List[List[float]], prompt: List[str]) -> Image:
        """Utility function to draw bbox on the image"""
        color_list = ['red', 'blue', 'yellow', 'purple', 'green', 'black', 'brown', 'orange', 'white', 'gray']
        width, height = pil_img.size
        draw = ImageDraw.Draw(pil_img)
        font_folder = os.path.dirname(os.path.dirname(__file__))
        font_path = os.path.join(font_folder, 'Rainbow-Party-2.ttf')
        font = ImageFont.truetype(font_path, 30)

        for box_id in range(len(bboxes)):
            obj_box = bboxes[box_id]
            text = prompt[box_id]
            fill = 'black'
            for color in prompt[box_id].split(' '):
                if color in color_list:
                    fill = color
            text = text.split(',')[0]
            x_min, y_min, x_max, y_max = (
                obj_box[0] * width,
                obj_box[1] * height,
                obj_box[2] * width,
                obj_box[3] * height,
            )
            draw.rectangle(
                [int(x_min), int(y_min), int(x_max), int(y_max)],
                outline=fill,
                width=4,
            )
            draw.text((int(x_min) + 20, int(y_min) + 10), text, fill=fill, font=font)

        return pil_img


    @torch.no_grad()
    def __call__(
            self,
            prompt: List[List[str]] = None,
            bboxes: List[List[List[float]]] = None,
            masks = None,
            depth = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            image: PipelineImageInput = None,
            ControlSteps=-1,
            aug_phase_with_and=False,
            sa_preserve=False,
            use_sa_preserve=False,
            clear_set=False,
            GUI_progress=None,
            refined_alpha=10.0,
            depth_img_token=None,
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            guess_mode: bool = False,
            control_guidance_start: Union[float, List[float]] = [0.0],
            control_guidance_end: Union[float, List[float]] = [1.0],
            control_fft_filter: bool = False,
            gamma_scale: float = 1.0,
            sam_masks=None,
            control_CN=False,
            fft_up=False,
            fft_mid=False,
            fft_up_res=8,
            fft_d=8
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            token_indices (Union[List[List[List[int]]], List[List[int]]], optional):
                The list of the indexes in the prompt to layout. Defaults to None.
            bboxes (Union[List[List[List[float]]], List[List[float]]], optional):
                The bounding boxes of the indexes to maintain layout in the image. Defaults to None.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            max_guidance_iter (`int`, *optional*, defaults to `10`):
                The maximum number of iterations for the layout guidance on attention maps in diffusion mode.
            max_guidance_iter_per_step (`int`, *optional*, defaults to `5`):
                The maximum number of iterations to run during each time step for layout guidance.
            scale_factor (`int`, *optional*, defaults to `50`):
                The scale factor used to update the latents during optimization.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        if self.use_controlnet:
            if image is None:
                raise ValueError("For controlnet, `image` must be passed.")
            controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        def aug_phase_with_and_function(phase, instance_num):
            instance_num = min(instance_num, 7)
            copy_phase = [phase] * instance_num
            phase = ', and '.join(copy_phase)
            return phase

        if aug_phase_with_and:
            instance_num = len(prompt[0]) - 1
            for i in range(1, len(prompt[0])):
                prompt[0][i] = aug_phase_with_and_function(prompt[0][i],
                                                            instance_num)
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_nums = [0] * len(prompt)
        for i, _ in enumerate(prompt):
            prompt_nums[i] = len(_)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if self.use_controlnet:

            global_pool_conditions = (
                controlnet.config.global_pool_conditions
                if isinstance(controlnet, ControlNetModel)
                else controlnet.nets[0].config.global_pool_conditions
            )
            guess_mode = guess_mode or global_pool_conditions
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
            
            

        # 3. Encode input prompt
        prompt_embeds, cond_prompt_embeds, embeds_pooler = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if self.use_controlnet:
            prompt_controlnet = [['masterpiece, best quality']]
            
            prompt_embeds_controlnet, cond_prompt_controlnet, embeds_pooler_controlnet = self._encode_prompt(
                prompt_controlnet,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                'worst quality, low quality, bad anatomy, watermark, text, blurry',
                prompt_embeds=None,
                negative_prompt_embeds=None,
            )
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        if self.use_controlnet:
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
        
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        if clear_set:
            self.instance_set = set()
            self.embedding = {}

        now_set = set()
        for i in range(len(bboxes[0])):
            now_set.add((tuple(bboxes[0][i]), prompt[0][i + 1]))

        mask_set = (now_set | self.instance_set) - (now_set & self.instance_set)
        self.instance_set = now_set

        if use_sa_preserve:
            guidance_mask = np.full((4, height // 8, width // 8), 1.0)
                    
            for bbox, _ in mask_set:
                w_min = max(0, int(width * bbox[0] // 8) - 5)
                w_max = min(width, int(width * bbox[2] // 8) + 5)
                h_min = max(0, int(height * bbox[1] // 8) - 5)
                h_max = min(height, int(height * bbox[3] // 8) + 5)
                guidance_mask[:, h_min:h_max, w_min:w_max] = 0
            
            kernal_size = 5
            guidance_mask = uniform_filter(
                guidance_mask, axes = (1, 2), size = kernal_size
            )
        
            guidance_mask = torch.from_numpy(guidance_mask).to(self.device).unsqueeze(0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # print(timesteps)
            for i, t in enumerate(timesteps):
                if GUI_progress is not None:
                    GUI_progress[0] = int((i + 1) / len(timesteps) * 100)
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                cross_attention_kwargs = {'prompt_nums': prompt_nums,
                                          'bboxes': bboxes,
                                          'masks': masks,
                                          'depth': depth,
                                          'ith': i,
                                          'embeds_pooler': embeds_pooler,
                                          'timestep': t,
                                          'height': height,
                                          'width': width,
                                          'ControlSteps': ControlSteps,
                                          'sa_preserve': sa_preserve,
                                          'use_sa_preserve': use_sa_preserve,
                                          'refined_alpha': refined_alpha,
                                          'depth_img_token': depth_img_token,
                                          'gamma_scale': gamma_scale,
                                          'sam_masks': sam_masks,
                                          }

                if self.use_controlnet:
                    # controlnet(s) inference
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds_controlnet
                    
                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]
                        
                    # print(image.shape) # torch.Size([2, 3, 456, 456])
                    # print(control_model_input.shape) # torch.Size([2, 4, 57, 57])
                    # print(controlnet_prompt_embeds.shape) # torch.Size([2, 77, 768])
                    
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds if control_CN else controlnet_prompt_embeds,
                        controlnet_cond=image,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                        cross_attention_kwargs=cross_attention_kwargs if control_CN else {}
                    )
                    
                # predict the noise residual
                
                self.unet.eval()
                if self.use_controlnet:
                    if control_fft_filter:
                        if fft_mid:
                            sup_mask = construct_supplement_mask_from_layout(bboxes[0], mid_block_res_sample.shape[2], mid_block_res_sample.shape[3])[0, 0, ...]
                            sup_mask = sup_mask.to(mid_block_res_sample.device)
                            mid_block_res_sample[:, :, sup_mask == 1] = Fourier_filter(mid_block_res_sample, fft_d=fft_d)[:, :, sup_mask == 1]

                        if fft_up:
                            after_filter = []
                            for i, block in enumerate(down_block_res_samples):
                                if block.shape[3] > fft_up_res:
                                    after_filter.append(block)
                                else:
                                    sup_mask = construct_supplement_mask_from_layout(bboxes[0], block.shape[2], block.shape[3])[0, 0, ...]
                                    sup_mask = sup_mask.to(mid_block_res_sample.device)
                                    block[:, :, sup_mask == 1] = Fourier_filter(block, fft_d=fft_d)[:, :, sup_mask == 1]
                                    after_filter.append(block)
                            down_block_res_samples = after_filter
                            
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples, 
                        # list, len = 12, 
                        mid_block_additional_residual=mid_block_res_sample,
                        # torch.Size([2, 1280, 8, 8])
                        # return_dict=False,
                    ).sample
                        
                else:
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs
                    ).sample
                    
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )

                step_output = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                )
                latents = step_output.prev_sample

                ori_input = latents.detach().clone()
                if use_sa_preserve and i in self.embedding:
                    latents = (
                            latents * (1.0 - guidance_mask)
                            + torch.from_numpy(self.embedding[i]).to(latents.device) * guidance_mask
                        ).float()
                
                if sa_preserve:
                    self.embedding[i] = ori_input.cpu().numpy()
        
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=None
        )


