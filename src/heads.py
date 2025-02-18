import os
import torch
from tqdm import tqdm

import open_clip

from src.datasets.templates import get_templates
from src.datasets.registry import get_dataset
from src.modeling import ClassificationHead, ImageEncoder


def build_classification_head(model, dataset_name, data_location, device, classnames):
    template = get_templates(dataset_name)

    if not classnames:
        classnames = get_dataset(dataset_name, None, location=data_location).classnames
    model.eval()
    model.to(device)

    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(device) # tokenize
            embeddings = model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= model.logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def build_subset_classification_head(model, dataset_name, classes, data_location, device,classnames=None):
    """
    template:主要通过生成与每个类相关的文本描述
    get_dataset:加载subset数据集
    """
    template = get_templates(dataset_name)
    # dataset = get_dataset(
    #     dataset_name,
    #     None,
    #     location=data_location
    # )
    if not classnames:
        classnames = get_dataset(dataset_name, None, location=data_location).classnames
    model.eval()
    model.to(device)
    
    print('Building SUBSET classification head.')
    with torch.no_grad():
        #存储zero-shot分类头的weight
        zeroshot_weights = []
        for class_idx in tqdm(classes):
            classname = classnames[class_idx]
            texts = []
            for t in template:
                texts.append(t(classname))
            #将文本描述列表转换为模型可以理解的格式（即将文本转换为模型的输入 token）asdsaaass
            texts = open_clip.tokenize(texts).to(device) # tokenize
            #使用文本编码器（通常是 CLIP 模型的文本部分）对文本进行编码，得到类的嵌入（embedding）。
            embeddings = model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= model.logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def get_classification_head(args, dataset_name, image_encoder=None, classnames=None):
    if isinstance(image_encoder, ImageEncoder) and image_encoder.has_lang():
        print('Using passed model to create classifier!')
        model = image_encoder.model
    else:
        model = ImageEncoder(args, keep_lang=True).model

    classification_head = build_classification_head(model, dataset_name, args.data_location, args.device, classnames)
    
    return classification_head
