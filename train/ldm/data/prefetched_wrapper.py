import numpy as np
import torch
import torch.distributed as dist


class PrefetchedWrapper(object):
    def prefetched_loader(loader):
        stream = torch.cuda.Stream()
        first = True

        for next_target in loader:
            with torch.cuda.stream(stream):
                next_target['pixel_values'] = next_target['pixel_values'].cuda(non_blocking=True)
                # next_input = next_input.float() / 0.5 - 1
                if 'input_ids' in next_target:
                    next_target['input_ids'] = next_target['input_ids'].cuda(non_blocking=True)
                if 'attention_mask' in next_target:
                    next_target['attention_mask'] = next_target['attention_mask'].cuda(non_blocking=True)
                if 'attn_mask' in next_target:
                    next_target['attn_mask'] = next_target['attn_mask'].cuda(non_blocking=True)
                if 'semantic_mask' in next_target:
                    next_target['semantic_mask'] = next_target['semantic_mask'].cuda(non_blocking=True)
                if 'box' in next_target:
                    next_target['box'] = next_target['box'].cuda(non_blocking=True)
                if 'phase_sup_mask' in next_target:
                    next_target['phase_sup_mask'] = next_target['phase_sup_mask'].cuda(non_blocking=True)
                if 'supplement_mask' in next_target:
                    next_target['supplement_mask'] = next_target['supplement_mask'].cuda(non_blocking=True)
                if 'true_text_mask' in next_target:
                    next_target['true_text_mask'] = next_target['true_text_mask'].cuda(non_blocking=True)
                if 'RGB_image' in next_target:
                    next_target['RGB_image'] = next_target['RGB_image'].cuda(non_blocking=True)
                if 'instance_img' in next_target:
                    next_target['instance_img'] = next_target['instance_img'].cuda(non_blocking=True)
                if 'instance_img_mask' in next_target:
                    next_target['instance_img_mask'] = next_target['instance_img_mask'].cuda(non_blocking=True)
                if 'instance_img_box' in next_target:
                    next_target['instance_img_box'] = next_target['instance_img_box'].cuda(non_blocking=True)
                if 'depth' in next_target:
                    next_target['depth'] = next_target['depth'].cuda(non_blocking=True)
                # dilated_mask
                if 'dilated_mask' in next_target:
                    next_target['dilated_mask'] = next_target['dilated_mask'].cuda(non_blocking=True)
            if not first:
                yield target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            target = next_target

        yield target

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0

    def __iter__(self):
        self.dataloader.sampler.set_epoch(self.epoch) # for local shuffle
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def fast_collate(batch, memory_format=torch.contiguous_format):
    imgs = [img[0] for img in batch]
    targets = torch.cat([target[1].unsqueeze(0) for target in batch], dim=0)
    w = imgs[0].size[0]
    h = imgs[0].size[1]

    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array.copy())

    return tensor, targets
