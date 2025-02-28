## Getting Started with ZoRI
Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.

###
Get modules ready before training:

###
To get refined text embedding indices, please run
```
python -m zori.utils.get_refined_channel --config configs/cache.yaml DATASET 'isaid_zsi_11_4_val'
```

###
To select visual channels, please run
```
python -m zori.utils.get_refined_vis_channel --config configs/cache.yaml DATASET 'isaid_zsi_11_4_train'
```

###
To prepare cache bank, please run
```
python -m zori.utils.cache_model --config configs/cache.yaml DATASET 'isaid_zsi_11_4_train_all' PROTOTYPE_NUM [4]
```

