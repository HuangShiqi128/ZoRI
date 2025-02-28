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
import itertools
import cv2
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts
)
from zori.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper

def get_indices(cfg, loader):
    
    # CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(cfg['MODEL_NAME'], pretrained=cfg['PRETRAIN'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for prop_num in cfg['PROTOTYPE_NUM']:
        with torch.no_grad():
            feats = [[] for _ in range(cfg['CATE_NUM'])]
            instance_count = {str(class_id): 0 for class_id in range(cfg['CATE_NUM'])}
            for i, batched_inputs in enumerate(tqdm(loader)):
                #print(f"Processing batch {i+1}")
                images = [x["image"].to(device) for x in batched_inputs]
                images = [(x - torch.Tensor(cfg['PIXEL_MEAN']).view(-1, 1, 1).cuda()) / torch.Tensor(cfg['PIXEL_STD']).view(-1, 1, 1).cuda() for x in images]
                instances = [x["instances"].to(device) for x in batched_inputs]
                for (image, instance) in zip(images, instances):
                    h, w = 320, 320
                    gt_boxes = instance.gt_boxes
                    # box_instances = []
                    for i, gt_box in enumerate(gt_boxes):
                        class_id = instance.gt_classes[i].item()
                        instance_count[str(class_id)] = instance_count.get(str(class_id), 0) + 1
                        if instance_count[str(class_id)] <= prop_num:
                            x_min, y_min, x_max, y_max = gt_box.int().tolist()
        
                            # Crop the image using the GT box coordinates
                            cropped_image = image[:, y_min:y_max, x_min:x_max]
                            resized_image = F.interpolate(cropped_image.unsqueeze(0), (h, w))
                            feat = model.visual.trunk.stem(resized_image)
                            feats[class_id].append(feat)
                if all(count >= prop_num for count in instance_count.values()):
                    break
            
            feats = list(itertools.chain(*feats))
            feats = torch.cat(feats, dim=0)
            feats /= feats.norm(dim=-1, keepdim=True) #[176, 192, 80, 80]
        
        feats = feats.reshape(cfg['CATE_NUM'], prop_num, 192, 80, 80)
        sim_sum = torch.zeros((192)).cuda()
        count = 0
        for i in range(cfg['CATE_NUM']):
            for j in range(cfg['CATE_NUM']):
                for m in range(prop_num):
                    for n in range(prop_num):
                        if i != j:
                            sim_sum += torch.sum(feats[i, m] * feats[j, n], dim=(1, 2))
                            count += 1
        sim = sim_sum / count
        
        criterion = (-1) * cfg['W'][0] * sim + cfg['W'][1] * torch.var(feats, dim=(0, 1, 3, 4))

        for channel_num in cfg['VIS_CHANNEL_NUM']:
            _, indices = torch.topk(criterion, k=channel_num)
            savefile = "{}/refined_channel_{}_{}_{}.pt".format(cfg['INDICES_DIR'], cfg['CATE_NUM'], prop_num, channel_num)
            torch.save(indices, savefile) 
    return indices


def main():

    # Load config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default = '', help='settings in yaml format')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cfg['INDICES_DIR'] = '{}/vis_indices'.format(cfg['OUTPUT_DIR'])
    os.makedirs(cfg['INDICES_DIR'], exist_ok=True)

    print("\nRunning configs.")
    print(cfg, "\n")

    torch.manual_seed(1)
    
    print("Preparing dataset.")
    
    # _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    dataset_name = cfg['DATASET']
    # Get metadata for the dataset
    metadata = MetadataCatalog.get(dataset_name)
    cfg['CATE_NUM'] = len(metadata.seen_index)
    # Get dataset dicts
    dataset_dicts = get_detection_dataset_dicts(dataset_name)
    train_loader = build_detection_test_loader(dataset=dataset_dicts, mapper=COCOInstanceNewBaselineDatasetMapper(cfg, image_format='RGB', tfm_gens=[]), batch_size=2)
 
    print("\nLoading visual features from train set.")
    indices = get_indices(cfg, train_loader)

if __name__ == '__main__':
    main()