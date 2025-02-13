import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import sys
import os
from .coco_utils import load_coco_json_zsi
import torch

# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
SIOR_ZSI_TRAIN_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "airplane"},
    # {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "airport"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "baseballfield"},
    # {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "basketballcourt"},
    {"color": [0, 0, 70], "isthing": 1, "id": 5, "name": "bridge"},
    {"color": [0, 0, 192], "isthing": 1, "id": 6, "name": "chimney"},
    {"color": [250, 0, 30], "isthing": 1, "id": 7, "name": "expressway-service-area"},
    {"color": [165, 42, 42], "isthing": 1, "id": 8, "name": "expressway-toll-station"},
    {"color": [182, 182, 255], "isthing": 1, "id": 9, "name": "dam"},
    {"color": [0, 82, 0], "isthing": 1, "id": 10, "name": "golffield"},
    # {"color": [199, 100, 0], "isthing": 1, "id": 11, "name": "groundtrackfield"},
    {"color": [72, 0, 118], "isthing": 1, "id": 12, "name": "harbor"},
    {"color": [255, 179, 240], "isthing": 1, "id": 13, "name": "overpass"},
    {"color": [209, 0, 151], "isthing": 1, "id": 14, "name": "ship"},
    {"color": [92, 0, 73], "isthing": 1, "id": 15, "name": "stadium"},
    {"color": [0, 228, 0], "isthing": 1, "id": 16, "name": "storagetank"},
    {"color": [145, 148, 174], "isthing": 1, "id": 17, "name": "tenniscourt"},
    {"color": [197, 226, 255], "isthing": 1, "id": 18, "name": "trainstation"},
    {"color": [9, 80, 61], "isthing": 1, "id": 19, "name": "vehicle"},
    # {"color": [84, 105, 51], "isthing": 1, "id": 20, "name": "windmill"}
]

SIOR_ZSI_TEST_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "airplane"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "airport"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "baseballfield"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "basketballcourt"},
    {"color": [0, 0, 70], "isthing": 1, "id": 5, "name": "bridge"},
    {"color": [0, 0, 192], "isthing": 1, "id": 6, "name": "chimney"},
    {"color": [250, 0, 30], "isthing": 1, "id": 7, "name": "expressway-service-area"},
    {"color": [165, 42, 42], "isthing": 1, "id": 8, "name": "expressway-toll-station"},
    {"color": [182, 182, 255], "isthing": 1, "id": 9, "name": "dam"},
    {"color": [0, 82, 0], "isthing": 1, "id": 10, "name": "golffield"},
    {"color": [199, 100, 0], "isthing": 1, "id": 11, "name": "groundtrackfield"},
    {"color": [72, 0, 118], "isthing": 1, "id": 12, "name": "harbor"},
    {"color": [255, 179, 240], "isthing": 1, "id": 13, "name": "overpass"},
    {"color": [209, 0, 151], "isthing": 1, "id": 14, "name": "ship"},
    {"color": [92, 0, 73], "isthing": 1, "id": 15, "name": "stadium"},
    {"color": [0, 228, 0], "isthing": 1, "id": 16, "name": "storagetank"},
    {"color": [145, 148, 174], "isthing": 1, "id": 17, "name": "tenniscourt"},
    {"color": [197, 226, 255], "isthing": 1, "id": 18, "name": "trainstation"},
    {"color": [9, 80, 61], "isthing": 1, "id": 19, "name": "vehicle"},
    {"color": [84, 105, 51], "isthing": 1, "id": 20, "name": "windmill"}
]

SIOR_ZSI_TEST_CATEGORIES_UNSEEN = [
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "airport"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "basketballcourt"},
    {"color": [199, 100, 0], "isthing": 1, "id": 11, "name": "groundtrackfield"},
    {"color": [84, 105, 51], "isthing": 1, "id": 20, "name": "windmill"}
]

unseen_classes = [
    "airport",
    "basketballcourt",
    "groundtrackfield",
    "windmill"
]

unseen_ids = [2, 4, 11, 20]
seen_ids = [1, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19]

def _get_sior_zsi_seen_instances_meta():
    thing_ids = [k["id"] for k in SIOR_ZSI_TRAIN_CATEGORIES]
    assert len(thing_ids) == 16, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SIOR_ZSI_TRAIN_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_all_sior16_4_instance_seen(root):
    metadata = _get_sior_zsi_seen_instances_meta()
    name = "sior_zsi_16_4_train"
    image_root = "/data/SIOR/train/images"
    json_file = "/data/SIOR/train/annotations/instances_train_seen_16_4.json"

    json_file = os.path.join(root, json_file) if "://" not in json_file else json_file
    image_root = os.path.join(root, image_root)
    DatasetCatalog.register(name, lambda: load_coco_json_zsi(json_file,
                                                                 image_root,
                                                                 name))

    MetadataCatalog.get(name).set(
            json_file=json_file,
            image_root=image_root,
            evaluation_set = unseen_classes,
            unseen_index=unseen_ids,
            seen_index=seen_ids,
            evaluator_type="coco_instance_gzero", **metadata)


def _get_sior_zsi_test_all_instances_meta():
    thing_ids = [k["id"] for k in SIOR_ZSI_TEST_CATEGORIES]
    assert len(thing_ids) == 20, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SIOR_ZSI_TEST_CATEGORIES]
    thing_colors = [k["color"] for k in SIOR_ZSI_TEST_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_all_sior16_4_instance_val_all(root):
    metadata = _get_sior_zsi_test_all_instances_meta()
    name = "sior_zsi_16_4_val"
    image_root = "/data/SIOR/val/images"
    json_file = "/data/SIOR/val/annotations/instances_val_gzsi.json"


    json_file = os.path.join(root,
                             json_file) if "://" not in json_file else json_file
    image_root = os.path.join(root, image_root)
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json_zsi(json_file,
                                                             image_root,
                                                             name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluation_set=unseen_classes,
        unseen_index=unseen_ids,
        seen_index=seen_ids,
        evaluator_type="coco_instance_gzero", **metadata)

def _get_sior_zsi_test_all_instances_meta():
    thing_ids = [k["id"] for k in SIOR_ZSI_TEST_CATEGORIES]
    assert len(thing_ids) == 20, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SIOR_ZSI_TEST_CATEGORIES]
    thing_colors = [k["color"] for k in SIOR_ZSI_TEST_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_sior16_4_instance_seen(_root)
register_all_sior16_4_instance_val_all(_root)








