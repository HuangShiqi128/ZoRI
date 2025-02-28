from tqdm import tqdm
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import open_clip
import os
import random
import argparse
import yaml
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper 
from detectron2.config import get_cfg
from collections import defaultdict

import cv2
from zori.data.datasets.register_isaid_zsi_11_4 import register_all_isaid11_4_instance_seen
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts
)
from zori.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper

import matplotlib.pyplot as plt
import numpy as np
from mmengine.visualization import Visualizer
import mmcv


def build_cache_model(cfg, loader):

    # CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(cfg['MODEL_NAME'], pretrained=cfg['PRETRAIN'])
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for num in cfg['PROTOTYPE_NUM']:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            sampled_instances = {str(class_id): [] for class_id in cfg['SEEN_IDS']}
            instance_count = {str(class_id): 0 for class_id in cfg['SEEN_IDS']}
            for i, batched_inputs in enumerate(tqdm(loader)):
                #print(f"Processing batch {i+1}")
                image_ids = [x["image_id"] for x in batched_inputs]
                images_ = [x["image"].to(device) for x in batched_inputs]
                images = [(x - torch.Tensor(cfg['PIXEL_MEAN']).view(-1, 1, 1).cuda()) / torch.Tensor(cfg['PIXEL_STD']).view(-1, 1, 1).cuda() for x in images_]
                instances = [x["instances"].to(device) for x in batched_inputs]
                for (image_, image_id, image, instance) in zip(images_, image_ids, images, instances):
                    h, w = 320, 320
                    image_ = torch.permute(image_, (1, 2, 0))
                    gt_boxes = instance.gt_boxes #XYXY
                    # box_instances = []
                    for i, gt_box in enumerate(gt_boxes):
                        class_id = instance.gt_classes[i].item()
                        if class_id in cfg['SEEN_IDS']:
                            if instance_count[str(class_id)] < num:
                                if image_id not in sampled_instances[str(class_id)]:
                                    instance_count[str(class_id)] += 1
                                    sampled_instances[str(class_id)].append(image_id)
                                    x_min, y_min, x_max, y_max = gt_box.int().tolist()
                
                                    # Crop the image using the GT box coordinates
                                    cropped_image = image[:, y_min:y_max, x_min:x_max]
                                    image_box = image_[y_min:y_max, x_min:x_max, :]
                                    
                                    if num==32:
                                        img_dir = '{}/imgs_{}'.format(cfg['OUTPUT_DIR'], num)
                                        os.makedirs(img_dir, exist_ok=True)
                                        fig, axs = plt.subplots(1, 2)
                                        axs[0].imshow(image_.int().cpu().numpy())
                                        axs[0].axis('off')
                                        axs[0].add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                                                linewidth=2, edgecolor='r', facecolor='none'))
                                        
                                        axs[1].imshow(image_box.int().cpu().numpy())
                                        axs[1].axis('off')

                                        plt.tight_layout()
                                        output_filename = f'img_{image_id}_class_{class_id}.jpg'
                                        plt.savefig("{}/{}".format(img_dir, output_filename))
                                        plt.close()
                                    
                                    resized_image = F.interpolate(cropped_image.unsqueeze(0), (h, w))
                                    cache_key = model.encode_image(resized_image)
                                    cache_keys.append(cache_key)
                                    cache_values.append(instance.gt_classes[i].unsqueeze(0))
            
            cache_keys = torch.cat(cache_keys, dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_values = F.one_hot(torch.cat(cache_values, dim=0))
            
            torch.save(cache_keys, cfg['OUTPUT_DIR'] + '/keys_' + str(num) + ".pt") #[192, 768]
            torch.save(cache_values, cfg['OUTPUT_DIR'] + '/values_' + str(num) + ".pt") #[192, 15]
    return


def main():

    # Load config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default = '', help='settings in yaml format')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    print("\nRunning configs.")
    print(cfg, "\n")

    torch.manual_seed(1)
    
    print("Preparing dataset.")
    
    # _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    # register_all_isaid_instance_train_all(_root)
    dataset_name = cfg['DATASET']
    # Get metadata for the dataset
    metadata = MetadataCatalog.get(dataset_name)
    cfg['SEEN_IDS'] = [i-1 for i in metadata.seen_index]
    # Get dataset dicts
    dataset_dicts = get_detection_dataset_dicts(dataset_name) #bbox XYWH
    train_loader = build_detection_test_loader(dataset=dataset_dicts, mapper=COCOInstanceNewBaselineDatasetMapper(cfg, image_format='RGB', tfm_gens=[]), batch_size=2)
    
    cfg['OUTPUT_DIR'] = '{}/seen_{}'.format(cfg['OUTPUT_DIR'], len(cfg['SEEN_IDS']))
    os.makedirs(cfg['OUTPUT_DIR'], exist_ok=True)

    print("\nLoading visual features from train set.")
    build_cache_model(cfg, train_loader)


if __name__ == '__main__':
    main()