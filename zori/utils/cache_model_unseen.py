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
import json
import matplotlib.pyplot as plt

import cv2
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts
)
from zori.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper

def build_cache_model(cfg, loader, pred_path):

    # CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(cfg['MODEL_NAME'], pretrained=cfg['PRETRAIN'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(pred_path, 'r') as f:
        pred = json.load(f)

    for num in cfg['PROTOTYPE_NUM']:
        cache_keys = []
        cache_values = []
        
        with torch.no_grad():
            top_preds = []
            top_image_ids = []
            
            for cat in cfg['UNSEEN_IDS']:
                filtered_pred = [annotation for annotation in pred if annotation['category_id'] == cat]

                # Sort filtered annotations by score in descending order
                sorted_pred = sorted(filtered_pred, key=lambda x: x['score'], reverse=True)

                selected_pred = []
                selected_image_ids = set()  # Set to keep track of selected image_ids

                for annotation in sorted_pred:
                    if annotation['image_id'] not in selected_image_ids and len(selected_pred) < num:
                        selected_pred.append(annotation)
                        selected_image_ids.add(annotation['image_id'])

                top_preds.extend(selected_pred)
                top_image_ids.extend(selected_image_ids)

                top_preds = sorted(top_preds, key=lambda x: x['image_id'])

            for i, batched_inputs in enumerate(tqdm(loader)):
                #print(f"Processing batch {i+1}")
                filtered_batched_inputs = [x for x in batched_inputs if x['image_id'] in top_image_ids]
                img_ids = [x['image_id'] for x in filtered_batched_inputs]
                images_ = [x["image"].to(device) for x in filtered_batched_inputs]
                images = [(x - torch.Tensor(cfg['PIXEL_MEAN']).view(-1, 1, 1).cuda()) / torch.Tensor(cfg['PIXEL_STD']).view(-1, 1, 1).cuda() for x in images_]
                instances = [x for x in top_preds if x['image_id'] in img_ids]
                for (image_, image, instance) in zip(images_, images, instances):
                    h_resize, w_resize = 320, 320
                    image_ = torch.permute(image_, (1, 2, 0))
                    boxes = instance['bbox']
                    x, y, w, h = [int(x) for x in boxes]
                    # Crop the image using the box coordinates
                    cropped_image = image[:, y:y+h, x:x+w]
                    image_box = image_[y:y+h, x:x+w, :]
                    
                    if num==1:
                        img_dir = '{}/imgs_{}'.format(cfg['OUTPUT_DIR'], num)
                        os.makedirs(img_dir, exist_ok=True)
                        fig, axs = plt.subplots(1, 2)
                        axs[0].imshow(image_.int().cpu().numpy())
                        axs[0].axis('off')
                        axs[0].add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none'))
                        axs[1].imshow(image_box.int().cpu().numpy())
                        axs[1].axis('off')

                        plt.tight_layout()
                        output_filename = "img_{}_class_{}.jpg".format(instance['image_id'], instance['category_id'])
                        plt.savefig("{}/{}".format(img_dir, output_filename))
                        plt.close() 
                    resized_image = F.interpolate(cropped_image.unsqueeze(0), (h_resize, w_resize))
                    cache_key = model.encode_image(resized_image)
                    cache_keys.append(cache_key)
                    cache_values.append(torch.tensor(instance['category_id']-1).unsqueeze(0))
                
            cache_keys = torch.cat(cache_keys, dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_values = F.one_hot(torch.cat(cache_values, dim=0), num_classes=15)
            
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
    
    prediction_path = cfg['RESULTS'] #results path
    # _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    dataset_name = cfg['DATASET']
    # Get metadata for the dataset
    metadata = MetadataCatalog.get(dataset_name)
    cfg['UNSEEN_IDS'] = metadata.unseen_index
    # Get dataset dicts
    dataset_dicts = get_detection_dataset_dicts(dataset_name)
    train_loader = build_detection_test_loader(dataset=dataset_dicts, mapper=COCOInstanceNewBaselineDatasetMapper(cfg, image_format='RGB', tfm_gens=[]), batch_size=1)

    cfg['OUTPUT_DIR'] = '{}/pseudo_unseen_{}'.format(cfg['OUTPUT_DIR'], len(cfg['UNSEEN_IDS']))
    os.makedirs(cfg['OUTPUT_DIR'], exist_ok=True)

    print("\nLoading visual features from val set.")
    build_cache_model(cfg, train_loader, prediction_path)


if __name__ == '__main__':
    main()