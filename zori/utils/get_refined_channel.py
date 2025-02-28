import torch
import open_clip
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts
)
import os
import argparse
import yaml
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from zori.data.datasets.register_isaid_zsi_11_4 import register_all_isaid11_4_instance_val_all
import numpy as np

RESISC45_PROMPT = [
    'satellite imagery of {}.',
    'aerial imagery of {}.',
    'satellite photo of {}.',
    'aerial photo of {}.',
    'satellite view of {}.',
    'aerial view of {}.',
    'satellite imagery of a {}.',
    'aerial imagery of a {}.',
    'satellite photo of a {}.',
    'aerial photo of a {}.',
    'satellite view of a {}.',
    'aerial view of a {}.',
    'satellite imagery of the {}.',
    'aerial imagery of the {}.',
    'satellite photo of the {}.',
    'aerial photo of the {}.',
    'satellite view of the {}.',
    'aerial view of the {}.',
]

def get_text_embeddings(cfg, class_names, clip_model, text_tokenizer, template):
    with torch.no_grad():
        clip_weights = []
        for classname in class_names:
            # Tokenize the prompts
            # classname = classname.replace('_', ' ')
            template_texts = [t.format(classname) for t in template]
            texts_token = text_tokenizer(template_texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda() #[768,11]
    return clip_weights


def get_indices(cfg, clip_weights):
    feat_dim, cate_num = clip_weights.shape
    text_feat = clip_weights.t().unsqueeze(1) #[15, 1, 768]
    feats = text_feat.squeeze() #[15, 768]
    
    sim_sum = torch.zeros((feat_dim)).cuda()
    count = 0
    for i in range(cate_num):
        for j in range(cate_num):
            if i != j:
                sim_sum += feats[i, :] * feats[j, :]
                count += 1
    sim = sim_sum / count
 
    criterion = (-1) * cfg['W'][0] * sim + cfg['W'][1] * torch.var(clip_weights, dim=1)
    
    for channel_num in cfg['CHANNEL_NUM']:
        _, indices = torch.topk(criterion, k=channel_num)
        savefile = "{}/refined_channel_{}.pt".format(cfg['INDICES_DIR'], channel_num)
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

    cfg['INDICES_DIR'] = '{}/text_indices'.format(cfg['OUTPUT_DIR'])
    os.makedirs(cfg['INDICES_DIR'], exist_ok=True)

    print("\nRunning configs.")
    print(cfg, "\n")

    torch.manual_seed(1)

    # _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    # register_all_isaid11_4_instance_val_all(_root)
    dataset_name = cfg['DATASET']
    metadata = MetadataCatalog.get(dataset_name)
    model, _, _ = open_clip.create_model_and_transforms(cfg['MODEL_NAME'], pretrained=cfg['PRETRAIN'])
    text_tokenizer = open_clip.get_tokenizer(cfg['MODEL_NAME'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompt_templates = RESISC45_PROMPT
    text_embeddings = get_text_embeddings(cfg, metadata.thing_classes, model, text_tokenizer, prompt_templates)

    indices = get_indices(cfg, text_embeddings)

if __name__ == '__main__':
    main()