import torch
import open_clip

from src import utils


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f'Loading {args.model} pre-trained weights.')
        if '__pretrained__' in args.model:
            name, pretrained = args.model.split('__pretrained__')
        else:
            name = args.model
            pretrained = 'openai'
        print(f'Loading pre-trained weights for model {name}, pretrained ckpt: {pretrained}.')
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir)
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)
    
    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)
        
    def has_lang(self) -> bool:
        return hasattr(self.model, 'transformer')
    
    def freeze_lang(self):
        for pg in self.model.transformer.parameters():
            pg.requires_grad = False

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return torch.load(filename)

    @classmethod
    def load_from_state_dict(cls, model_name, state_dict, pretrained='openai'):
        """
        根据给定的 state_dict 加载模型参数并重新初始化 ImageEncoder 实例。

        :param model_name: 模型名称
        :param state_dict: 加载的模型参数字典
        :param pretrained: 使用的预训练权重（默认为 'openai'）
        :return: 初始化后的 ImageEncoder 实例
        """
        # 创建模型和预处理操作
        model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cls.cache_dir
        )

        # 将给定的 state_dict 加载到模型中
        model.load_state_dict(state_dict)

        # 创建并返回新的 ImageEncoder 实例
        encoder = cls.__new__(cls)  # 使用 __new__ 创建实例（不调用 __init__）
        encoder.model = model
        encoder.train_preprocess = train_preprocess
        encoder.val_preprocess = val_preprocess
        encoder.cache_dir = cls.cache_dir  # 如果需要，可以设置其他属性
        return encoder

    # @classmethod
    # def load_from_state_dict(cls, model_name, state_dict):
    #     self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
    #         name, pretrained=pretrained, cache_dir=args.openclip_cachedir)
    #     self.model.load_from_state_dict(state_dict)
        

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


def concat_classification_heads(clsf_heads):
    normalizes = [clsf_head.normalize for clsf_head in clsf_heads]
    assert len(set(normalizes)) == 1  # check if the parameter is equal for all

    ws = [clsf_head.weight for clsf_head in clsf_heads]
    w = torch.cat(ws, dim=0)
    
    bs = [clsf_head.bias for clsf_head in clsf_heads]
    b = torch.cat(bs, dim=0)
    
    return ClassificationHead(normalizes[0], w, b)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    """
    在迁移学习中，通常会冻结大部分网络的参数，只对最后的分类头部分进行微调。这样可以减少训练的时间，同时避免改变预训练模型中学到的有用特征。
    确保在训练过程中不更新分类头的权重和偏置。这样做通常是为了避免在微调过程中修改分类头的参数，尤其是在迁移学习时，您可能希望保持已经预训练好的分类头不变。
    """
    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)
    """
    这个方法的目的是冻结图像编码器（image_encoder）的语言相关部分（通常是与文本嵌入、语言模型或者多模态学习中的语言部分相关的模块）。
    这意味着，调用 freeze_lang() 会将 image_encoder 中与语言相关的部分的参数冻结，不参与训练过程。
    freeze_lang() 可能会冻结图像编码器中的文本嵌入部分。这样做是为了防止在训练过程中修改已经预训练好的语言部分（如文本编码器的权重）。
    """
    def freeze_lang(self):
        self.image_encoder.freeze_lang()

    def freeze(self):
        self.freeze_head()
        for pg in self.image_encoder.parameters():
            pg.requires_grad = False

    def forward(self, inputs, return_features=False):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs if not return_features else (outputs, features)

    def __call__(self, inputs, return_features=False):
        return self.forward(inputs, return_features)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)


class MultiHeadImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_heads):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = torch.nn.ModuleList(classification_heads)
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward(self, inputs, head_idx):
        features = self.image_encoder(inputs)
        outputs = self.classification_heads[head_idx](features)
        return outputs

    def __call__(self, inputs, head_idx):
        return self.forward(inputs, head_idx)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)
