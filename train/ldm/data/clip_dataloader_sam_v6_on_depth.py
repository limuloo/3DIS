import io
import os
import os.path
import zipfile

import lmdb
import torch
import warnings
from PIL import Image, ImageFile
import random
import re
import pickle
import numpy as np
from functools import partial
# design own zipfile pattern, helpful for memory offload
# from .slim_zipfile import SlimZipFile as ZipFile
import math
import torch.distributed as dist
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms

from .prefetched_wrapper import PrefetchedWrapper, fast_collate
# from ..tokenizer.bpe import get_encoder
# from ..common import tokenize
from transformers import BertTokenizer, BertModel
# from .deborder import hasWhiteBorder, captionDetect
import tarfile
import json
from typing import Union, List
from pycocotools import mask as mask_utils
import torch.nn.functional as F
import cv2
# import albumentations as A



warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['CLIP_zip_dataloader']



def get_input_ids(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding = "max_length",
        max_length = tokenizer.model_max_length,
        truncation = True,
        return_tensors = "pt"
    )
    return text_inputs.input_ids


def make_zip_fns(data_list, this_card_all_idx):
    zip_files = []
    for zip_idx in this_card_all_idx:
        zip_path = data_list.replace("part_idx.pkl", "part{:04d}.zip".format(zip_idx))
        zip_files.append(zip_path)
    return zip_files


def deserialize(serialized_data):
    deserialized_data = pickle.loads(serialized_data)
    return deserialized_data


class FeaDatasetListZip(VisionDataset):
    
    def del_color_from_prompt(self, prompt):
        color_list = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'pink', 'brown', 'gray', 'grey']
        for color in color_list:
            if color in prompt:
                print(f'del color: {color}')
            prompt = prompt.replace(color, '')
            
        return prompt
    
    def read_from_tar(self, tar_path: str, file_name: Union[str, int]):
        if tar_path[-1].isdigit():  # lmdb type
            tar_id = int(os.path.split(tar_path)[-1])
            if self.tar_fns[tar_id] is None:
                self.tar_fns[tar_id] = lmdb.open(tar_path + self.lmdb_suffix, readonly=True, lock=False, readahead=False, meminit=False)
                assert not self.use_blip_caption, "Now, you are using COCO2014 Dataset with its captions, and you don't need BLIP caption."
                for json_id in range(tar_id * self.lmdb2json_ratio, (tar_id + 1) * self.lmdb2json_ratio):
                    # print(self.segmentation_label_root)
                    json_path = os.path.join(self.segmentation_label_root, f'{str(json_id).zfill(6)}.json')
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                        self.segmentation_label_dict = {**self.segmentation_label_dict, **json_data}
            # import pdb; pdb.set_trace()
            # print(str(file_name).encode('ascii'))
            with self.tar_fns[tar_id].begin() as txn:
                value = txn.get(str(file_name).encode('ascii'))
            datum = deserialize(value)
            jpg_name = datum['jpg_name']
            if self.load_RGB:
                if self.RGB_lmdb is None:
                    self.RGB_lmdb = lmdb.open(self.RGB_path, readonly=True)
                with self.RGB_lmdb.begin() as txn:
                    RGB_value = txn.get(jpg_name.encode())
                    RGB_img = Image.open(io.BytesIO(RGB_value))
            # 特判，有些图片可能因为try块 没有对应的标注
            # 0000021452.jpg
            # print(len(self.segmentation_label_dict.keys()))
            # print(jpg_name)
            segment_label = self.segmentation_label_dict[jpg_name[: -4] + '.jpg']
            if type(segment_label) != dict:
                segment_label_len = len(segment_label)
                choose_caption = random.randint(0, segment_label_len - 1)
                segment_label = segment_label[choose_caption]
            if 'caption' in segment_label:
                description = segment_label['caption']
            elif 'desc' in datum:
                description = datum['desc']['caption']
            else:
                assert False
            if 'annotation' in segment_label:
                segment_label = segment_label['annotation']
            image = datum['feature']
            if self.del_color:
                description = self.del_color_from_prompt(description)
        else:
            assert False
        return_dict = {
            'vae_feature': image,
            'prompt': description,
            'segment_label': segment_label,
            'jpg_name': jpg_name # only for debug
        }
        if self.load_RGB:
            return_dict['RGB_image'] = RGB_img
        return return_dict

    def probe_in_tar(self, tar_file_path: str):
        '''
        输入: filename是tar的路徑。
        返回: 该tar下所有以'.jpg'为后缀的文件名。
        '''
        if tar_file_path[-1].isdigit():
            # lmdb
            print(tar_file_path + self.lmdb_suffix)
            with lmdb.open(tar_file_path + self.lmdb_suffix, readonly=True, lock=False, readahead=False, meminit=False) as lmdb_f:
                with lmdb_f.begin() as txn:
                    return range(txn.stat()['entries'])
        else:
            tar_file_id = int(os.path.split(tar_file_path)[-1][:-4])
            tar_file_suffix = tar_file_path[-4:]
            if tar_file_suffix == '.tar':
                with tarfile.open(tar_file_path, 'r') as f:
                    names = f.getnames()
                    jpg_names = [o for o in names if o.endswith('.pkl')]  # 只留.pkl的文件
            elif tar_file_suffix == '.zip':
                with zipfile.ZipFile(tar_file_path, 'r') as f:
                    names = f.namelist()
                    jpg_names = [o for o in names if o.endswith('.pkl')]  # 只留.pkl的文件
            else:
                assert False, "Now, only support .zip or .tar!"
        return jpg_names

    def make_tar_dataset(self, path_template, this_card_all_idx):
        '''
        输入: path_template为模板路径, this_card_all_idx为根据卡划分的id。
        返回: 一个list, 包含所有的{jpg_name, }
        '''
        imgs_desc = []
        tar_paths = [path_template.replace("idx", "{:06d}".format(tar_idx)) for tar_idx in this_card_all_idx]
        for tar_path in tar_paths:
            # 读取tar文件
            pkl_names = self.probe_in_tar(tar_path)
            imgs_desc.extend([(pkl_name, tar_path) for pkl_name in pkl_names])
        print("total data num: {}".format(len(imgs_desc)))
        return imgs_desc
    
    
    def __init__(self, list_data_root, data_list, caption_shuffle_percent, local_shuffle_type,
                 zip_max_split, mode, tokenizer, mask_ratio, transform=None, target_transform=None, context_length=77,
                 random_crop=False, random_flip=True, resolution=64, deborder=False, idx_len=0,
                 use_bbox=False, bbox_root='', bbox_thredshold=0.0,
                 use_bbox_caption_aug=False, bbox_class_dict_path='', bbox_caption_size_limit=20,
                 use_blip_caption=False, blip_caption_root='', blip_caption_p=1.0, segmentation_label_root='',
                 mask_padding=10, lmdb2json_ratio=100, limit_dataloader_len=-1,
                 attn_mask_neg_inf=False, attn_mask_softmax=False, attn_mask_amplify=1.0, phase_num=5,
                 only_use_one_box=False, desc_use_sup_mask=False,
                 swap_desc_with_null=False, swap_desc_with_null_p=0.0, null_use_one_box=False,
                 drop_desc=False, mask_area_threshold=0.0, append_uncond=False, 
                 phase_random_order=False, phase_random_order_v2=False, one_phase_one_instance=False,
                 phase_random_before_anything=False, phase_random_order_v3=False, bbox_mask_dilation=False,
                 phase_random_order_v4=False, cat_small_size=False, phase_random_order_v3_p=1.0,
                 use_limit_instance_num=False, limit_instance_num=1000, replace_desc_with_phase=False,
                 aug_phase_with_and=False, replace_desc_with_phase_p=1.0, inter_mode='nearest',
                 load_RGB=False, RGB_path='', instance_score_threshold=0.0, train_with_mask=False, replace_box_with_mask_p=0.0,
                 dilate_mask=False, dilate_kernel=15, dilate_iter=5, dilate_p=0.5,
                 drop_phase_in_img=False, drop_phase_in_img_p=0.5, replace_desc_with_null=False, replace_desc_with_null_p=0.0, lmdb_suffix='', del_color=False, uniform_aug_phase_with_and=False,
                 use_json_v2=False, replace_phase_with_null=False, min_edge_limit=-1):
        super(FeaDatasetListZip, self).__init__(list_data_root, transform=transform, target_transform=target_transform)
        # self.caption_shuffle_percent = caption_shuffle_percent 这个参数目前没有任何用
        # this_card_all_idx: local shuffle mode or others
        print(f'[Dataset Initialization Info] RANK{os.environ["RANK"]}')
        self.use_blip_caption = use_blip_caption
        self.lmdb2json_ratio = lmdb2json_ratio
        self.limit_dataloader_len = limit_dataloader_len
        self.attn_mask_neg_inf = attn_mask_neg_inf
        self.attn_mask_softmax = attn_mask_softmax
        self.attn_mask_amplify = attn_mask_amplify
        self.phase_num = phase_num
        self.drop_desc = drop_desc
        self.append_uncond = append_uncond
        self.mask_area_threshold = mask_area_threshold
        self.dilate_mask = dilate_mask
        self.dilate_kernel = dilate_kernel
        self.dilate_p = dilate_p
        self.dilate_iter = dilate_iter
        self.only_use_one_box = only_use_one_box
        self.desc_use_sup_mask = desc_use_sup_mask
        self.null_use_one_box = null_use_one_box
        self.swap_desc_with_null = swap_desc_with_null
        self.swap_desc_with_null_p = swap_desc_with_null_p
        self.phase_random_order = phase_random_order
        self.phase_random_order_v2 = phase_random_order_v2
        self.phase_random_order_v3 = phase_random_order_v3
        self.phase_random_order_v3_p = phase_random_order_v3_p
        self.use_limit_instance_num = use_limit_instance_num
        self.limit_instance_num = limit_instance_num
        self.replace_desc_with_phase = replace_desc_with_phase
        self.replace_desc_with_phase_p = replace_desc_with_phase_p
        self.aug_phase_with_and = aug_phase_with_and
        self.phase_random_order_v4 = phase_random_order_v4
        self.one_phase_one_instance = one_phase_one_instance
        self.phase_random_before_anything = phase_random_before_anything
        self.bbox_mask_dilation= bbox_mask_dilation
        self.cat_small_size = cat_small_size
        self.inter_mode = inter_mode
        self.load_RGB = load_RGB
        self.RGB_path = RGB_path
        self.RGB_lmdb = None
        self.instance_score_threshold = instance_score_threshold
        self.train_with_mask = train_with_mask
        self.replace_box_with_mask_p = replace_box_with_mask_p
        self.uniform_aug_phase_with_and = uniform_aug_phase_with_and
        self.drop_phase_in_img = drop_phase_in_img
        self.drop_phase_in_img_p = drop_phase_in_img_p
        self.replace_desc_with_null = replace_desc_with_null
        self.replace_desc_with_null_p = replace_desc_with_null_p
        self.lmdb_suffix = lmdb_suffix
        self.del_color = del_color
        self.min_edge_limit = min_edge_limit
        self.filter_list = ["top", "side", "a view", "various types", "front", "half", "the side", "lots", "a picture",
                            "a close up", "a close up", "a pair", "a group", "to her", "A group", "A couple", "A view"]
        self.use_json_v2 = use_json_v2
        self.replace_phase_with_null = replace_phase_with_null
        if self.use_blip_caption:
            self.blip_caption_root = blip_caption_root
            self.blip_caption_dict = {}
            self.blip_caption_p = blip_caption_p

        ###################################################### ADD
        self.segmentation_label_root = segmentation_label_root
        self.segmentation_label_dict = {}

        if idx_len != 0:
            self.this_card_all_idx = range(idx_len)
        else:
            self.this_card_all_idx = range(10)  # 需要用來黨訓練集的tar的id。

        self.tar_fns: List[Union[None, tarfile.TarFile, ZipFile], lmdb.Environment] = [None] * len(
            self.this_card_all_idx)
        self.samples = self.make_tar_dataset(list_data_root, self.this_card_all_idx)
        # list_data_root, tar的路徑模板
        # self.samples是一個列表，每個item是（圖片的名稱+對應的tar路徑）,例如('0000017589.pkl', '/home/zdw/data/LAION_SUPER_MINI/000001.tar')。

        # self.samples = make_dataset(list_data_root, self.this_card_all_idx)
        # self.zip_files = make_zip_fns(data_list, self.this_card_all_idx)
        self.context_length = context_length  # 77, 標準設定
        self.use_bbox = use_bbox
        self.bbox_root = bbox_root
        self.bbox_thredshold = bbox_thredshold
        # self.max_size = len(self.samples)
        # print(list_data_root, data_list)
        # print(self.max_size)
        self.deborder = deborder  # True, 但是還不知道這個具體是幹什麽的
        ######## for finetuning ################
        self.mode = mode  # 'train'
        ########################################
        self.do_tokenization = True
        self.random_crop = random_crop  # False
        self.random_flip = random_flip  # True
        self.resolution = resolution  # 512
        self.tokenizer = tokenizer  # CLIP Tokenizer
        self.mask_ratio = mask_ratio  # 0.1
        self.use_bbox_caption_aug = use_bbox_caption_aug
        self.bbox_caption_size_limit = bbox_caption_size_limit
        
        if self.use_bbox_caption_aug:
            self.bbox_class_dict = []
            with open(bbox_class_dict_path, 'r') as f:
                self.bbox_class_dict = json.load(f)
        self.mask_padding = mask_padding

    # def init_zip_fns(self, zip_idx):
    #     # for more details, https://discuss.pytorch.org/t/dataloader-with-zipfile-failed/42795
    #     if self.zip_fns[zip_idx] is None:
    #         self.zip_fns[zip_idx] = ZipFile(self.zip_files[zip_idx])

    def bbox_caption_size_check(self, full_shape, bbox):
        full_area = full_shape[0] * full_shape[1]
        now_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return now_area * self.bbox_caption_size_limit >= full_area

    def dilate_bbox(self, segment_bbox):
        W1, H1, W2, H2 = segment_bbox
        shift_range = 10
        W1 = max(0, (W1 - random.randint(0, shift_range)))
        H1 = max(0, (H1 - random.randint(0, shift_range)))
        W2 = min(512, (W2 + random.randint(0, shift_range)))
        H2 = min(512, (H2 + random.randint(0, shift_range)))
        return (W1, H1, W2, H2)


    def get_img_desc(self, index):
        try_time = 0
        while True:
            try_time += 1
            if try_time >= 100:
                assert False, "Bad!"
            try:
                # zip_idx: every GPU map to several zip, "zip_idx" will help point out the right zip
                jpg_idx, tar_idx = self.samples[
                    index]  # for example, '0000003674.jpg' and '/home/zdw/data/LAION_SUPER_MINI/000000.tar'
                # 接着，利用jpg_idx 与 tar_idx 从tar中读取文件
                o = self.read_from_tar(tar_idx, jpg_idx)  # 需要優化
                # print(666)
                sample = o['vae_feature']
                desc = o['prompt']
                seg_label = o['segment_label']
                jpg_name = o['jpg_name']
                if self.load_RGB:
                    RGB_img = o['RGB_image']
                    if RGB_img.mode != 'RGB':
                        RGB_img = RGB_img.convert('RGB')
                    RGB_img = np.array(RGB_img)
                    RGB_img = torch.from_numpy(RGB_img)
                    RGB_img = RGB_img.permute(2, 0, 1)
                    RGB_img = RGB_img[None, ...]
                    RGB_img = F.interpolate(RGB_img, size=(512, 512))
                    RGB_img = RGB_img[0, ...]
                    RGB_img = (RGB_img / 255.).float().numpy()  # (3, 512, 512)
                phase_num = len(seg_label.keys())
                phase_list = []
                semantic_mask_list = []  # bounding box 对应的 01 mask
                dilated_mask_list = []
                if self.load_RGB:
                    instance_img_list = []  # 每个instance 对应的图 (224, 224, 3)
                    instance_img_mask_list = []
                    instance_img_box_list = []
                box_list = []
                phase_book = set()
                if self.drop_phase_in_img and np.random.uniform() < self.drop_phase_in_img_p:
                    assert False
                            
                if self.one_phase_one_instance:
                    for key in seg_label.keys():
                        # 如果使用new_json的话len(seg_label.keys())只会为1
                        if self.use_json_v2:
                            # print('begin checking')
                            assert len(seg_label.keys()) == 1, f'if use_json_v2, len(seg_label.keys()) should be 1, but len(seg_label.keys())={len(seg_label.keys())}'
                        
                        segment_data = seg_label[key]
                        current_phase = ' '.join(segment_data['labels'][0].split())
                        # print(current_phase, )
                        if current_phase in self.filter_list and not self.use_json_v2:
                            continue
                        if current_phase in phase_book: # 如果使用new_json的话, 一定不走这个, 所以不用担心
                            continue
                        phase_book.add(current_phase)
                        # 'a drawing 0.55' ---> 'a drawing'
                        if 'bbox' in segment_data and \
                                segment_data['bbox'] is not None and \
                                len(segment_data['bbox']) > 0:
                            segment_bboxes = segment_data['bbox']
                            if self.load_RGB:
                                instance_shapes = segment_data['segmentation']
                            if self.train_with_mask:
                                instance_shapes = segment_data['segmentation']
                            if self.only_use_one_box and len(segment_bboxes) > 1:
                                assert False, "Currently, you set only_use_one_box=True, so we pass this sample"
                            for i, segment_bbox in enumerate(segment_bboxes):
                                # instance_score = float(segment_data['labels'][i].split()[-1])
                                # if instance_score < self.instance_score_threshold:
                                    # print('Filter!!!', segment_data['labels'][i])
                                    # continue
                                semantic_mask = np.zeros((1, 512, 512))
                                # clip W&H to 0 ~ 511
                                W1, H1, W2, H2 = segment_bbox
                                W1 = max(0, min(512, W1))
                                W2 = max(0, min(512, W2))
                                H1 = max(0, min(512, H1))
                                H2 = max(0, min(512, H2))
                                
                                assert W1 <= W2 and H1 <= H2
                                width = W2 - W1
                                height = H2 - H1
                                min_edge = min(width, height)

                                semantic_mask[:, int(H1): int(H2), int(W1): int(W2)] = 1

                                if self.train_with_mask and np.random.uniform() <= self.replace_box_with_mask_p:
                                    semantic_mask = instance_shapes[i]
                                    semantic_mask = mask_utils.decode(semantic_mask)[None, ...]
                                
                                if self.dilate_mask and np.random.uniform() < self.dilate_p:
                                    if type(self.dilate_iter) == int:
                                        dilated = cv2.dilate(semantic_mask[0, ...], np.ones((self.dilate_kernel, self.dilate_kernel), np.uint8), iterations=self.dilate_iter)
                                    else:
                                        iter = random.choice(self.dilate_iter)
                                        dilated = cv2.dilate(semantic_mask[0, ...], np.ones((self.dilate_kernel, self.dilate_kernel), np.uint8), iterations=iter)
                                    dilated = dilated[None, ...]
                                else:
                                    dilated = semantic_mask
                                
                                instance_img_mask = np.zeros((1, 512, 512))
                                if self.load_RGB:
                                    instance_shape = instance_shapes[i]
                                    instance_shape = mask_utils.decode(instance_shape)[None, ...]
                                    instance_img = RGB_img * instance_shape  # (3, 512, 512)
                                    instance_img = instance_img + np.ones((3, 512, 512)) * (1 - instance_shape)
                                    instance_img = instance_img[:, int(H1): int(H2), int(W1): int(W2)]
                                    instance_img = torch.from_numpy(instance_img)[None, ...]
                                    instance_img = F.interpolate(instance_img, (512, 512))
                                    instance_img = instance_img[0, ...].numpy()
                                    instance_img_mask[:, int(H1): int(H2), int(W1): int(W2)] = 1
                                    if self.train_with_mask and np.random.uniform() <= 0.5:
                                        instance_img_mask = instance_shape
                                if semantic_mask.sum() / (512 * 512) >= self.mask_area_threshold and min_edge >= self.min_edge_limit:
                                    if self.load_RGB:
                                        instance_img_list.append(instance_img)
                                        instance_img_mask_list.append(instance_img_mask)
                                        instance_img_box_list.append(np.array([[W1, W2, H1, H2]]))
                                    
                                    if self.del_color:
                                        phase_list.append(self.del_color_from_prompt(segment_data['labels'][i]))
                                    else:
                                        phase_list.append(segment_data['labels'][i])
                                    if self.replace_phase_with_null:
                                        phase_list[-1] = ""
                                    dilated_mask_list.append(dilated)
                                    semantic_mask_list.append(semantic_mask)
                                    box_list.append(np.array([[W1, W2, H1, H2]]))
                                    if self.use_limit_instance_num and len(phase_list) > self.limit_instance_num:
                                        assert False, f"You set limit_instance_num as {self.limit_instance_num}, and this sample over it!"
                # if len(phase_list) == 0:
                #     assert  False
                else:
                    assert False
                # gt_mask = torch.from_numpy(gt_mask).float()
                # 讀出來一個圖片和caption
                if desc == None:
                    # Sometimes caption is not exist in .pkl file, so we choose to set it as a empty string.
                    desc = ''
                # self.init_zip_fns(zip_idx)
                # with self.zip_fns[zip_idx].open(str(key), 'r') as f:
                #     sample = Image.open(f).convert('RGB')
                if self.mode == 'finetune':
                    mask_desc = torch.rand(1) > 1 - self.mask_ratio
                    if mask_desc:
                        desc = ''
                if self.swap_desc_with_null and np.random.uniform() < self.swap_desc_with_null_p:
                    desc = ''
                ret_result = {}
                ret_result['sample'] = sample
                ret_result['desc'] = desc
                ret_result['phase_list'] = phase_list
                ret_result['semantic_mask_list'] = semantic_mask_list
                ret_result['dilated_mask_list']= dilated_mask_list
                ret_result['box_list'] = box_list
                ret_result['jpg_name'] = jpg_name
                if self.load_RGB:
                    ret_result['RGB_image'] = RGB_img
                    ret_result['instance_img_list'] = instance_img_list
                    ret_result['instance_img_mask_list'] = instance_img_mask_list
                    ret_result['instance_img_box_list'] = instance_img_box_list
                return ret_result
            except Exception as e:
            
                # solve missing image case.
                old_index = index
                index = index + random.randint(1, 9)
                index = max(0, index)
                _, zip_idx = self.samples[old_index]
                if index >= len(self.samples) - 1:
                    # for last index fail case.
                    index = 0
                # print('Error!', e.args, str(e))
                # print("Warning: load zip_idx={}-{} fail, change index {}->{} ".format(zip_idx, index, old_index, index))

    def get_sup_mask(self, mask_list):
        or_mask = np.zeros((1, 512, 512))
        for mask in mask_list:
            or_mask += mask
        or_mask[or_mask >= 1] = 1
        sup_mask = 1 - or_mask
        return sup_mask

    def aug_phase_with_and_function(self, phase, phase_num):
        copy_phase = [phase] * phase_num
        phase = ', and '.join(copy_phase)
        return phase

    def __getitem__(self, index):
        data = self.get_img_desc(index)
        sample = data['sample']
        desc = data['desc']
        phase_list = data['phase_list']
        jpg_name = data['jpg_name']
        semantic_mask_list = data['semantic_mask_list']
        dilated_mask_list = data['dilated_mask_list']
        box_list = data['box_list']
        if self.load_RGB:
            RGB_img = data['RGB_image']
            instance_img_list = data['instance_img_list']
            instance_img_mask_list = data['instance_img_mask_list']
            instance_img_box_list = data['instance_img_box_list']
            assert len(instance_img_list) == len(instance_img_mask_list)
            assert len(phase_list) == len(instance_img_list)
        assert len(phase_list) == len(semantic_mask_list)
        assert len(phase_list) == len(box_list)
        if self.cat_small_size:
            for i in range(len(phase_list)):
                semantic_mask = semantic_mask_list[i]
                _, H, W = semantic_mask.shape
                if semantic_mask.sum() / (H * W) <= 0.09:
                    phase_list[i] = phase_list[i] + ", small size"

        if self.phase_random_before_anything:
            # 这个和phase_random_v2的区别是在截断前打乱，可以让数据更多样。
            random_idx = list(range(len(phase_list)))
            random.shuffle(random_idx)
            phase_list = [phase_list[o] for o in random_idx]
            semantic_mask_list = [semantic_mask_list[o] for o in random_idx]
            dilated_mask_list = [dilated_mask_list[o] for o in random_idx]
            if self.load_RGB:
                instance_img_list = [instance_img_list[o] for o in random_idx]
                instance_img_mask_list = [instance_img_mask_list[o] for o in random_idx]
                instance_img_box_list = [instance_img_box_list[o] for o in random_idx]
                
            box_list = [box_list[o] for o in random_idx]

        if self.aug_phase_with_and:
            true_phase_num = len(phase_list)
            if self.uniform_aug_phase_with_and:
                phase_list = [self.aug_phase_with_and_function(o, random.randint(1, 6)) for o in phase_list]
            else:
                phase_list = [self.aug_phase_with_and_function(o, true_phase_num) for o in phase_list]

        if self.drop_desc:
            phase_list = phase_list
            semantic_mask_list = semantic_mask_list
            # if self.load_RGB:
            #     instance_img_list = instance_img_list
            box_list = box_list
        else:
            if self.replace_desc_with_phase and len(phase_list) > 0 and np.random.uniform() < self.replace_desc_with_phase_p:
                desc = phase_list[random.randint(0, min(len(phase_list), self.phase_num - 1) - 1)]
            if self.replace_desc_with_null and np.random.uniform() < self.replace_desc_with_null_p:
                desc = ''
            phase_list = [desc] + phase_list
            if self.desc_use_sup_mask:
                # if self.load_RGB:
                #     instance_img_list = [RGB_img] + instance_img_list
                semantic_mask_list = [self.get_sup_mask(semantic_mask_list)] + semantic_mask_list
                dilated_mask_list = [self.get_sup_mask(dilated_mask_list)] + dilated_mask_list
                box_list = [np.array([[0, 512, 0, 512]])] + box_list
            else:
                # if self.load_RGB:
                #     instance_img_list = [RGB_img] + instance_img_list
                semantic_mask_list = [np.ones((1, 512, 512))] + semantic_mask_list
                dilated_mask_list = [np.ones((1, 512, 512))] + dilated_mask_list
                box_list = [np.array([[0, 512, 0, 512]])] + box_list

        sup_mask = self.get_sup_mask(semantic_mask_list[1:])

        ##################  补全或者截断  ##################
        if self.phase_num > len(phase_list):
            add_num = self.phase_num - len(phase_list)
            phase_list += ["" for i in range(add_num)]
            if self.null_use_one_box:
                if self.load_RGB:
                    instance_img_list += [np.ones((3, 512, 512)) for i in range(add_num)]
                    instance_img_mask_list += [np.zeros((1, 512, 512)) for i in range(add_num)]
                semantic_mask_list += [np.ones((1, 512, 512)) for i in range(add_num)]
                dilated_mask_list += [np.ones((1, 512, 512)) for i in range(add_num)]
                box_list += [np.array([[0, 512, 0, 512]]) for i in range(add_num)]
            else:
                if self.load_RGB:
                    instance_img_list += [np.ones((3, 512, 512)) for i in range(add_num)]
                    instance_img_mask_list += [np.zeros((1, 512, 512)) for i in range(add_num)]
                semantic_mask_list += [np.zeros((1, 512, 512)) for i in range(add_num)]
                dilated_mask_list += [np.zeros((1, 512, 512)) for i in range(add_num)]
                box_list += [np.array([[0, 0, 0, 0]]) for i in range(add_num)]
        else:
            phase_list = phase_list[: self.phase_num]
            semantic_mask_list = semantic_mask_list[: self.phase_num]
            dilated_mask_list = dilated_mask_list[: self.phase_num]
            if self.load_RGB:
                instance_img_list = instance_img_list[: self.phase_num]
                instance_img_mask_list = instance_img_mask_list[: self.phase_num]
            box_list = box_list[: self.phase_num]
        if self.phase_random_order:
            random_idx = list(range(self.phase_num))
            random.shuffle(random_idx)
            phase_list = [phase_list[o] for o in random_idx]
            semantic_mask_list = [semantic_mask_list[o] for o in random_idx]
            dilated_mask_list = [dilated_mask_list[o] for o in random_idx]
            if self.load_RGB:
                instance_img_list = [instance_img_list[o] for o in random_idx]
                instance_img_mask_list = [instance_img_mask_list[o] for o in random_idx]
            box_list = [box_list[o] for o in random_idx]
        if self.phase_random_order_v2:
            # phase_random_v2主要是想打乱短语和空文本的顺序, 全局文本不变的。
            assert not self.phase_random_order
            assert not self.drop_desc
            random_idx = list(range(self.phase_num - 1))
            random.shuffle(random_idx)
            random_idx = [0] + [(o + 1) for o in random_idx]
            phase_list = [phase_list[o] for o in random_idx]
            semantic_mask_list = [semantic_mask_list[o] for o in random_idx]
            dilated_mask_list = [dilated_mask_list[o] for o in random_idx]
            if self.load_RGB:
                instance_img_list = [instance_img_list[o] for o in random_idx]
                instance_img_mask_list = [instance_img_mask_list[o] for o in random_idx]
            box_list = [box_list[o] for o in random_idx]
        if self.phase_random_order_v3:
            assert not self.phase_random_order
            assert not self.phase_random_order_v2
            assert not self.drop_desc
            if np.random.uniform() < self.phase_random_order_v3_p:
                random_idx = list(range(self.phase_num - 1))
                random.shuffle(random_idx)
                random_idx = [random.randint(1, self.phase_num - 1)] + [(o + 1) for o in random_idx]
                phase_list = [phase_list[o] for o in random_idx]
                semantic_mask_list = [semantic_mask_list[o] for o in random_idx]
                dilated_mask_list = [dilated_mask_list[o] for o in random_idx]
                if self.load_RGB:
                    instance_img_list = [instance_img_list[o] for o in random_idx]
                    instance_img_mask_list = [instance_img_mask_list[o] for o in random_idx]
                box_list = [box_list[o] for o in random_idx]
        if self.phase_random_order_v4:
            assert not self.phase_random_order
            assert not self.phase_random_order_v2
            assert not self.phase_random_order_v3
            assert not self.drop_desc
            random_idx = list(range(self.phase_num - 1))
            random.shuffle(random_idx)
            random_idx = [random.randint(0, self.phase_num - 1)] + [(o + 1) for o in random_idx]
            phase_list = [phase_list[o] for o in random_idx]
            semantic_mask_list = [semantic_mask_list[o] for o in random_idx]
            dilated_mask_list = [dilated_mask_list[o] for o in random_idx]
            if self.load_RGB:
                instance_img_list = [instance_img_list[o] for o in random_idx]
                instance_img_mask_list = [instance_img_mask_list[o] for o in random_idx]
            box_list = [box_list[o] for o in random_idx]


        if self.append_uncond:
            phase_list.append("")
            semantic_mask_list.append(sup_mask)
            dilated_mask_list.append(sup_mask)
            # if self.load_RGB:
            #     instance_img_list.append(np.zeros(3, 512, 512))
            box_list.append(np.array([[0, 512, 0, 512]]))


        # attn_mask = attn_mask * self.attn_mask_amplify
        # if self.attn_mask_neg_inf:
        #     attn_mask[attn_mask == 0] = float('-inf')
        # if self.attn_mask_softmax:
        #     attn_mask = F.softmax(attn_mask, dim=0)

        if self.deborder:  # 這個應該是去白邊的，先不用吧
            if hasWhiteBorder(sample):
                caption_h = captionDetect(sample)
                sample = sample.crop((0, 0, sample.size[0], sample.size[1] - caption_h))

        if self.transform is not None:
            arr = self.transform(sample)
        else:
            arr = sample


        if self.target_transform is not None:
            try:
                desc = self.target_transform(desc)
                model_kwargs = dict(y=desc)
            except:
                print("Descripion is unavailale to transform", desc)
        phase_sup_mask = torch.zeros((self.phase_num, self.phase_num))
        if self.do_tokenization:
            true_text_mask = torch.zeros((len(phase_list), ))
            input_ids = []
            attention_mask = []
            captions = []
            for i, desc in enumerate(phase_list):
                tokens = self.tokenizer(desc, max_length=self.context_length,
                                        padding="do_not_pad", truncation=True).input_ids
                mask = self.tokenizer.pad({"input_ids": tokens}, padding="max_length",
                                          return_tensors="pt", max_length=self.context_length)
                input_ids.append(get_input_ids(tokenizer=self.tokenizer, prompt=desc))
                attention_mask.append(mask.attention_mask[None, :])
                captions.append(desc)
                if desc != "":
                    true_text_mask[i] = 1

            for i in range(self.phase_num):
                for j in range(self.phase_num):
                    if phase_list[i] != phase_list[j]:
                        phase_sup_mask[i, j] = 1

            semantic_mask_list = [torch.from_numpy(o).float() for o in semantic_mask_list]
            semantic_mask_list = [F.interpolate(o[None, ...], (64, 64), mode=self.inter_mode)[0, ...] for o in semantic_mask_list]
            semantic_mask = torch.cat(semantic_mask_list, dim=0)
            dilated_mask_list = [torch.from_numpy(o).float() for o in dilated_mask_list]
            dilated_mask_list = [F.interpolate(o[None, ...], (64, 64), mode=self.inter_mode)[0, ...] for o in dilated_mask_list]
            dilated_mask = torch.cat(dilated_mask_list, dim=0)
            
            # semantic_mask = torch.from_numpy(semantic_mask)

            if self.load_RGB:
                RGB_img = torch.from_numpy(RGB_img)

            # instance_img_transfrom = A.Compose([
            #     A.Resize(height=224, width=224),
            #     A.HorizontalFlip(p=0.5),
            #     A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, value=(1, 1, 1)),
            #     A.Blur(p=0.3),
            #     A.ElasticTransform(p=0.3, border_mode=cv2.BORDER_CONSTANT, value=(1, 1, 1))
            # ])

            if self.load_RGB:
                instance_img_list = [instance_img_transfrom(image=o.transpose(1, 2, 0))['image'] for o in instance_img_list]
                instance_img_list = [o.transpose(2, 0, 1) for o in instance_img_list]
                instance_img_list = [torch.from_numpy(o).float()[None, ...] for o in instance_img_list]
                instance_img = torch.cat(instance_img_list, dim=0)
                instance_img_mask_list = [torch.from_numpy(o).float() for o in instance_img_mask_list]
                instance_img_mask_list = [F.interpolate(o[None, ...], (64, 64), mode=self.inter_mode)[0, ...] for o in instance_img_mask_list]
                instance_img_mask = torch.cat(instance_img_mask_list, dim=0)
            # instance_img = F.interpolate(instance_img, (224, 224), mode=self.inter_mode)

            supplement_mask = torch.from_numpy(sup_mask).float()
            supplement_mask = F.interpolate(supplement_mask[None, ...], (64, 64), mode=self.inter_mode)[0, ...]

            box_list = [torch.from_numpy(o).float() / 512. for o in box_list]
            box = torch.cat(box_list, dim=0)

            input_ids = torch.cat(input_ids, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)
            ret = {
                "pixel_values": arr,  # VAE feature
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "captions": captions,
                "semantic_mask": semantic_mask,
                "dilated_mask": dilated_mask,
                "box": box,
                "phase_sup_mask": phase_sup_mask,
                "supplement_mask": supplement_mask,
                "true_text_mask": true_text_mask,
                "jpg_name": jpg_name
            }
            if self.load_RGB:
                ret['RGB_image'] = RGB_img
                ret['instance_img'] = instance_img
                ret['instance_img_mask'] = instance_img_mask
            return ret
        else:
            return arr, model_kwargs

    def __len__(self):
        if self.limit_dataloader_len != -1 and len(self.samples) >= self.limit_dataloader_len:
            return self.limit_dataloader_len
        return len(self.samples)
        # return 8000

    def set_max_size(self, max_size=None):
        if max_size is not None:
            self.max_size = max_size

    def __del__(self):
        print('delete all self.tar_fns')
        for tar_fn in self.tar_fns:
            if tar_fn is not None:
                tar_fn.close()


# @DATALOADER_REGISTRY.register
def CLIP_fea_zip_dataloader(cfg, tokenizer):
    # todo, ugly code for local_shuffle_type, devil figures; consider only support zip format :)
    assert cfg.local_shuffle_type in [0, 4]

    def _caption_transform(caption):
        pat = re.compile(r'[#]+')
        return re.sub(pat, ' ', caption)

    dataset = FeaDatasetListZip(list_data_root=os.path.join(cfg.list_data_root, cfg.data_list),
                                data_list=None,
                                transform=None,
                                mode=cfg.mode,
                                target_transform=_caption_transform,
                                context_length=cfg.context_length,
                                caption_shuffle_percent=cfg.caption_shuffle_percent,
                                local_shuffle_type=cfg.local_shuffle_type,
                                zip_max_split=cfg.zip_max_split,
                                tokenizer=tokenizer,
                                mask_ratio=cfg.mask_ratio,
                                resolution=cfg.resolution,
                                deborder=cfg.deborder,
                                idx_len=cfg.idx_len,
                                use_bbox=cfg.use_bbox,
                                bbox_root=cfg.bbox_root,
                                bbox_thredshold=cfg.bbox_thredshold,
                                use_bbox_caption_aug=cfg.use_bbox_caption_aug,
                                bbox_class_dict_path=cfg.bbox_class_dict_path,
                                use_blip_caption=cfg.use_blip_caption,
                                blip_caption_root=cfg.blip_caption_root,
                                blip_caption_p=cfg.blip_caption_p,
                                segmentation_label_root=cfg.segmentation_label_root,
                                mask_padding=cfg.mask_padding,
                                lmdb2json_ratio=cfg.lmdb2json_ratio,
                                limit_dataloader_len=cfg.limit_dataloader_len,
                                attn_mask_neg_inf=cfg.attn_mask_neg_inf,
                                attn_mask_softmax=cfg.attn_mask_softmax,
                                attn_mask_amplify=cfg.attn_mask_amplify,
                                phase_num=cfg.phase_num,
                                drop_desc=cfg.drop_desc,
                                mask_area_threshold=cfg.mask_area_threshold,
                                append_uncond=cfg.append_uncond,
                                # dilate_mask=cfg.dilate_mask,
                                # dilate_kernel=cfg.dilate_kernel,
                                # dilate_iter=cfg.dilate_iter,
                                only_use_one_box=cfg.only_use_one_box,
                                desc_use_sup_mask=cfg.desc_use_sup_mask,
                                swap_desc_with_null=cfg.swap_desc_with_null,
                                swap_desc_with_null_p=cfg.swap_desc_with_null_p,
                                null_use_one_box=cfg.null_use_one_box,
                                phase_random_order=cfg.phase_random_order,
                                phase_random_order_v2=cfg.phase_random_order_v2,
                                one_phase_one_instance=cfg.one_phase_one_instance,
                                phase_random_before_anything=cfg.phase_random_before_anything,
                                phase_random_order_v3=cfg.phase_random_order_v3,
                                bbox_mask_dilation=cfg.bbox_mask_dilation,
                                phase_random_order_v4=cfg.phase_random_order_v4,
                                cat_small_size=cfg.cat_small_size,
                                phase_random_order_v3_p=cfg.phase_random_order_v3_p,
                                use_limit_instance_num=cfg.use_limit_instance_num,
                                limit_instance_num=cfg.limit_instance_num,
                                replace_desc_with_phase=cfg.replace_desc_with_phase,
                                replace_desc_with_phase_p=cfg.replace_desc_with_phase_p,
                                aug_phase_with_and=cfg.aug_phase_with_and,
                                inter_mode=cfg.inter_mode,
                                load_RGB=cfg.load_RGB,
                                RGB_path=cfg.RGB_path,
                                instance_score_threshold=cfg.instance_score_threshold,
                                train_with_mask=cfg.train_with_mask,
                                replace_box_with_mask_p=cfg.replace_box_with_mask_p,
                                uniform_aug_phase_with_and=cfg.uniform_aug_phase_with_and,
                                drop_phase_in_img=cfg.drop_phase_in_img,
                                drop_phase_in_img_p=cfg.drop_phase_in_img_p,
                                replace_desc_with_null=cfg.replace_desc_with_null,
                                replace_desc_with_null_p=cfg.replace_desc_with_null_p,
                                lmdb_suffix=cfg.lmdb_suffix,
                                del_color=cfg.del_color,
                                dilate_mask=cfg.dilate_mask,
                                dilate_p=cfg.dilate_p,
                                dilate_kernel=cfg.dilate_kernel,
                                dilate_iter=cfg.dilate_iter,
                                use_json_v2=cfg.use_json_v2,
                                replace_phase_with_null=cfg.replace_phase_with_null,
                                min_edge_limit=cfg.min_edge_limit
                                )

    tensor = torch.zeros(cfg.world_size).cuda()
    tensor[cfg.local_rank] = len(dataset)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    min_dataset_length = int(tensor.min().item())
    print("rank{}, set dataset size from {} to {}".format(cfg.local_rank, len(dataset), min_dataset_length))
    dataset.set_max_size(min_dataset_length)

    # if cfg.local_shuffle_type == 4:
    #     sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=1, rank=0)
    # else:
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(dataset, shuffle=True)
    # 因爲目前是單機調試，暫時用裸的sampler

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.train_batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, sampler=sampler,
        drop_last=True, persistent_workers=True)  # 把原來的timeout=3600去掉

    loader = PrefetchedWrapper(loader)
    return loader







