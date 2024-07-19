import os
from collections import deque, defaultdict
import datetime
import time

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
import torchvision.transforms as T
from torchvision.datasets import CocoDetection


def get_transform():
    transforms = [T.ToTensor()]
    return torchvision.transforms.Compose(transforms)


def extract_probs_from_scores(scores):
    mean = torch.mean(scores)
    std = torch.std(scores)
    standardized_scores = (scores - mean) / std
    return torch.sigmoid(standardized_scores)


def compute_coco_areas(coco):
    ann_ids = coco.getAnnIds()
    for ann_id in ann_ids:
        ann = coco.loadAnns(ann_id)[0]
        if 'segmentation' not in ann or not ann['segmentation']:
            continue
        annotation = ann['segmentation'][0]
        segmentation = np.reshape(np.array(annotation, dtype=np.float32), (-1, 2))
        area = cv2.contourArea(segmentation)
        ann['area'] = area
    return coco


def threshold_keypoints(pred_keypoints, keypoints_scores, threshold=0.5):
    thresholded_keypoints = torch.zeros((20,3))
    for i, score in enumerate(keypoints_scores):
        if score > threshold:
            thresholded_keypoints[i] = torch.tensor(pred_keypoints[i])
    return thresholded_keypoints


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)


    def add_meter(self, name, meter):
        self.meters[name] = meter


    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


class COCODataset(CocoDetection):
    def __getitem__(self, idx):
        img, target = super(COCODataset, self).__getitem__(idx)
        img_id = self.ids[idx]
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        bbox_yxhw = torch.tensor(anno[0]['bbox']).reshape(1, 4)
        imgs = self.coco.loadImgs(ids=[img_id])

        target = {
            'image_id': img_id,
            'img_name': imgs[0]['file_name'],
            'boxes': torchvision.ops.box_convert(bbox_yxhw, 'xywh', 'xyxy'),
            'labels': torch.tensor([1 for _ in range(1)], dtype=torch.int64),
            'keypoints': torch.tensor(anno[0]['keypoints']).reshape(1, -1, 3),
        }
        return img, target


def plot_keypoints(image, keypoints, pred_keypoints, bbox, pred_bbox, dest_dir, image_id):
    fig, ax = plt.subplots(1)
    fig.set_size_inches(8, 8)
    ax.imshow(image)
    plt.axis('off')

    ## Draws real bbox (green)
    x1, y1, w, h = torchvision.ops.box_convert(bbox, 'xyxy', 'xywh')
    rect = plt.Rectangle((x1, y1), w, h, fill=False, color='green')
    ax.add_patch(rect)

    ## Draws predicted bbox (red)
    x1, y1, w, h = torchvision.ops.box_convert(pred_bbox, 'xyxy', 'xywh')
    rect = plt.Rectangle((x1, y1), w, h, fill=False, color='red')
    ax.add_patch(rect)

    ## Draws real keypoints (green)
    for j, keypoint in enumerate(keypoints):
        x, y, v = keypoint
        circle = plt.Circle((x, y), 2, color='green', fill=True)
        ax.add_patch(circle)

    ## Draws predicted keypoints (green)
    for j, keypoint in enumerate(pred_keypoints):
        x, y, v = keypoint
        circle = plt.Circle((x, y), 2, color='red', fill=True)
        ax.add_patch(circle)

    image_name = f'pred_{image_id}'
    plt.savefig(os.path.join(dest_dir, image_name))
    plt.close()
