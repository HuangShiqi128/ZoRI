import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import sys
import os
from .coco_utils import load_coco_json_zsi
import torch

# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
ISAID_ZSI_TRAIN_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "ship"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "storage_tank"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "baseball_diamond"},
    # {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "tennis_court"},
    {"color": [0, 0, 70], "isthing": 1, "id": 5, "name": "basketball_court"},
    {"color": [0, 0, 192], "isthing": 1, "id": 6, "name": "Ground_Track_Field"},
    {"color": [250, 0, 30], "isthing": 1, "id": 7, "name": "Bridge"},
    {"color": [165, 42, 42], "isthing": 1, "id": 8, "name": "Large_Vehicle"},
    {"color": [182, 182, 255], "isthing": 1, "id": 9, "name": "Small_Vehicle"},
    # {"color": [0, 82, 0], "isthing": 1, "id": 10, "name": "Helicopter"},
    # {"color": [199, 100, 0], "isthing": 1, "id": 11, "name": "Swimming_pool"},
    {"color": [72, 0, 118], "isthing": 1, "id": 12, "name": "Roundabout"},
    # {"color": [255, 179, 240], "isthing": 1, "id": 13, "name": "Soccer_ball_field"},
    {"color": [209, 0, 151], "isthing": 1, "id": 14, "name": "plane"},
    {"color": [92, 0, 73], "isthing": 1, "id": 15, "name": "Harbor"},
]

ISAID_ZSI_TEST_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "ship"},
    {"color": [0, 82, 0], "isthing": 1, "id": 2, "name": "storage_tank"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "baseball_diamond"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "tennis_court"},
    {"color": [0, 0, 70], "isthing": 1, "id": 5, "name": "basketball_court"}, 
    {"color": [0, 0, 192], "isthing": 1, "id": 6, "name": "Ground_Track_Field"},
    {"color": [199, 100, 0], "isthing": 1, "id": 7, "name": "Bridge"}, #[250, 0, 30]
    {"color": [0, 0, 142], "isthing": 1, "id": 8, "name": "Large_Vehicle"},
    {"color": [182, 182, 255], "isthing": 1, "id": 9, "name": "Small_Vehicle"},
    {"color": [0, 82, 0], "isthing": 1, "id": 10, "name": "Helicopter"},
    {"color": [255, 179, 240], "isthing": 1, "id": 11, "name": "Swimming_pool"},
    {"color": [72, 0, 118], "isthing": 1, "id": 12, "name": "Roundabout"},
    {"color": [255, 179, 240], "isthing": 1, "id": 13, "name": "Soccer_ball_field"},
    {"color": [209, 0, 151], "isthing": 1, "id": 14, "name": "plane"},
    {"color": [92, 0, 73], "isthing": 1, "id": 15, "name": "Harbor"},
]

ISAID_ZSI_TEST_CATEGORIES_UNSEEN = [
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "tennis_court"},
    {"color": [0, 82, 0], "isthing": 1, "id": 10, "name": "Helicopter"},
    {"color": [199, 100, 0], "isthing": 1, "id": 11, "name": "Swimming_pool"},
    {"color": [255, 179, 240], "isthing": 1, "id": 13, "name": "Soccer_ball_field"},
  
]


unseen_classes = [
    "tennis_court",
    "Helicopter",
    "Swimming_pool",
    "Soccer_ball_field"
]

unseen_ids = [4, 10, 11, 13]
seen_ids = [1, 2, 3, 5, 6, 7, 8, 9, 12, 14, 15]

def _get_isaid_zsi_seen_instances_meta():
    thing_ids = [k["id"] for k in ISAID_ZSI_TRAIN_CATEGORIES]
    assert len(thing_ids) == 11, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in ISAID_ZSI_TRAIN_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_all_isaid11_4_instance_seen(root):
    metadata = _get_isaid_zsi_seen_instances_meta()
    name = "isaid_zsi_11_4_train"
    image_root = "/data/iSAID_patches/train/images"
    json_file = "/data/iSAID_patches/train/annotations/instances_train_seen_11_4.json"

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


def _get_isaid_zsi_test_all_instances_meta():
    thing_ids = [k["id"] for k in ISAID_ZSI_TEST_CATEGORIES]
    assert len(thing_ids) == 15, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in ISAID_ZSI_TEST_CATEGORIES]
    thing_colors = [k["color"] for k in ISAID_ZSI_TEST_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_all_isaid11_4_instance_val_all(root):
    metadata = _get_isaid_zsi_test_all_instances_meta()
    name = "isaid_zsi_11_4_val"
    image_root = "/data/iSAID_patches/val/images"
    json_file = "/data/iSAID_patches/val/annotations/instances_val_gzsi.json"


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

def register_all_isaid11_4_instance_train_all(root):
    metadata = _get_isaid_zsi_test_all_instances_meta()
    name = "isaid_zsi_11_4_train_all"
    image_root = "/data/iSAID_patches/train/images"
    json_file = "/data/iSAID_patches/train/annotations/instancesonly_filtered_train.json"


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
    
def _get_isaid_zsi_test_unseen_instances_meta():
    thing_ids = [k["id"] for k in ISAID_ZSI_TEST_CATEGORIES_UNSEEN]
    assert len(thing_ids) == 4, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in ISAID_ZSI_TEST_CATEGORIES_UNSEEN]
    thing_colors = [k["color"] for k in ISAID_ZSI_TEST_CATEGORIES_UNSEEN]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_all_isaid11_4_instance_val_unseen(root):
    metadata = _get_isaid_zsi_test_unseen_instances_meta()
    name = "isaid_zsi_11_4_val_unseen"
    image_root = "/data/iSAID_patches/val/images"
    json_file = "/data/iSAID_patches/val/annotations/instances_val_unseen_11_4.json"

    json_file = os.path.join(root,
                             json_file) if "://" not in json_file else json_file
    image_root = os.path.join(root, image_root)

    DatasetCatalog.register(name, lambda: load_coco_json_zsi(json_file,
                                                             image_root,
                                                             name))

    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluation_set=unseen_classes,
        unseen_index=unseen_ids,
        seen_index=seen_ids,
        evaluator_type="coco_instance_gzero", **metadata)

def _get_isaid_zsi_test_all_instances_meta():
    thing_ids = [k["id"] for k in ISAID_ZSI_TEST_CATEGORIES]
    assert len(thing_ids) == 15, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in ISAID_ZSI_TEST_CATEGORIES]
    thing_colors = [k["color"] for k in ISAID_ZSI_TEST_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_isaid11_4_instance_seen(_root)
register_all_isaid11_4_instance_val_all(_root)
register_all_isaid11_4_instance_train_all(_root)
register_all_isaid11_4_instance_val_unseen(_root)








