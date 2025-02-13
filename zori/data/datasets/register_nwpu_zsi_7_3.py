import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import sys
import os
from .coco_utils import load_coco_json_zsi
import torch

# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
NWPU_ZSI_TRAIN_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "airplane"},
    # {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "ship"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "storage_tank"},
    {"color": [255, 179, 240], "isthing": 1, "id": 4, "name": "baseball_diamond"},
    {"color": [0, 0, 70], "isthing": 1, "id": 5, "name": "tennis_court"},
    # {"color": [92, 0, 73], "isthing": 1, "id": 6, "name": "basketball_court"},
    {"color": [250, 0, 30], "isthing": 1, "id": 7, "name": "ground_track_field"},
    # {"color": [165, 42, 42], "isthing": 1, "id": 8, "name": "harbor"},
    {"color": [182, 182, 255], "isthing": 1, "id": 9, "name": "bridge"},
    {"color": [0, 82, 0], "isthing": 1, "id": 10, "name": "vehicle"}
]

NWPU_ZSI_TEST_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "airplane"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "ship"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "storage_tank"},
    {"color": [255, 179, 240], "isthing": 1, "id": 4, "name": "baseball_diamond"},
    {"color": [0, 0, 70], "isthing": 1, "id": 5, "name": "tennis_court"},
    {"color": [92, 0, 73], "isthing": 1, "id": 6, "name": "basketball_court"},
    {"color": [250, 0, 30], "isthing": 1, "id": 7, "name": "ground_track_field"},
    {"color": [165, 42, 42], "isthing": 1, "id": 8, "name": "harbor"},
    {"color": [182, 182, 255], "isthing": 1, "id": 9, "name": "bridge"},
    {"color": [0, 82, 0], "isthing": 1, "id": 10, "name": "vehicle"}
]

NWPU_ZSI_TEST_CATEGORIES_UNSEEN = [
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "ship"},
    {"color": [92, 0, 73], "isthing": 1, "id": 6, "name": "basketball_court"},
    {"color": [165, 42, 42], "isthing": 1, "id": 8, "name": "harbor"}
  
]

unseen_classes = [
    "ship",
    "basketball_court",
    "harbor"
]

unseen_ids = [2, 6, 8]
seen_ids = [1, 3, 4, 5, 7, 9, 10]

def _get_nwpu_zsi_seen_instances_meta():
    thing_ids = [k["id"] for k in NWPU_ZSI_TRAIN_CATEGORIES]
    assert len(thing_ids) == 7, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in NWPU_ZSI_TRAIN_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_all_nwpu7_3_instance_seen(root):
    metadata = _get_nwpu_zsi_seen_instances_meta()
    name = "nwpu_zsi_7_3_train"
    image_root = "/data/NWPU/images"
    json_file = "/data/NWPU/annotations/instances_train_seen_7_3.json"

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


def _get_nwpu_zsi_test_all_instances_meta():
    thing_ids = [k["id"] for k in NWPU_ZSI_TEST_CATEGORIES]
    assert len(thing_ids) == 10, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in NWPU_ZSI_TEST_CATEGORIES]
    thing_colors = [k["color"] for k in NWPU_ZSI_TEST_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_all_nwpu7_3_instance_val_all(root):
    metadata = _get_nwpu_zsi_test_all_instances_meta()
    name = "nwpu_zsi_7_3_val"
    image_root = "/data/NWPU/images"
    json_file = "/data/NWPU/annotations/instances_val_gzsi.json"


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

def register_all_nwpu7_3_instance_train_all(root):
    metadata = _get_nwpu_zsi_test_all_instances_meta()
    name = "nwpu_zsi_7_3_train_all"
    image_root = "/data/NWPU/images"
    json_file = "/data/NWPU/annotations/instances_train.json"


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
        seen_index=seen_ids,
        evaluator_type="coco_instance_gzero", **metadata)
    
def _get_nwpu_zsi_test_unseen_instances_meta():
    thing_ids = [k["id"] for k in NWPU_ZSI_TEST_CATEGORIES_UNSEEN]
    assert len(thing_ids) == 3, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in NWPU_ZSI_TEST_CATEGORIES_UNSEEN]
    thing_colors = [k["color"] for k in NWPU_ZSI_TEST_CATEGORIES_UNSEEN]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_all_nwpu7_3_instance_val_unseen(root):
    metadata = _get_nwpu_zsi_test_unseen_instances_meta()
    name = "nwpu_zsi_7_3_val_unseen"
    image_root = "/data/NWPU/images"
    json_file = "/data/NWPU/annotations/instances_val_unseen_7_3.json"

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

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_nwpu7_3_instance_seen(_root)
register_all_nwpu7_3_instance_val_all(_root)
register_all_nwpu7_3_instance_train_all(_root)
register_all_nwpu7_3_instance_val_unseen(_root)








