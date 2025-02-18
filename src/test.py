import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from src.datasets.common import maybe_dictionarize
from src.eval import evaluate
from src.modeling import ImageClassifier
from src.heads import get_classification_head
from src.utils import get_logits



class Localizer(nn.Module):
    """
    trainable_params: dict key(name of struct of model ) and value (tensor)
    model : model after finetune
    pretrained_model: model before finetune
    finetuned_model : model after finetune (copy.deepcopy(model))
    后者都是用来计算得到tv的
    """
    def __init__(self, trainable_params, model, pretrained_model, finetuned_model, dataset_name, args, graft_args, classifier_head,model_type="roberta"):
        super(Localizer, self).__init__()

        self.params = trainable_params
        self.model = model
        self.pretrained_model = pretrained_model
        self.finetuned_model = finetuned_model
        self.args = args
        self.graft_args = graft_args
        self.model_type = model_type

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
        self.model.to(self.device)
        if self.model_type == "vit":
            print("Using Vision Transformer Classifier Head")
            self.classifier_head = classifier_head if classifier_head is not None else get_classification_head(self.args, dataset_name)
            self.classifier_head.to(self.device)

        self.pretrained_model.to("cpu")
        self.finetuned_model.to("cpu")

        self.model.eval()
        self.finetuned_model.eval()
        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False   
        for param in self.finetuned_model.parameters():
            param.requires_grad = False
        # Move tensors to the appropriate device
        self.pretrained_state_dict = {key: value.to(self.device) for key, value in pretrained_model.state_dict().items()}

        # 将 finetuned_state_dict 中的每个 tensor 移动到正确的设备
        self.finetuned_state_dict = {key: value.to(self.device) for key, value in finetuned_model.state_dict().items()}
        self.model_state_dict = {key: value.to(self.device) for key, value in model.state_dict().items()}
        
        self.create_binary_masks()
        self.mask = self.create_basepatch()


    def create_binary_masks(self):
        self.trainable_name = []
        self.trainable_parameters = []
        # 对于每个参数，创建一个与其相同形状的新张量
        for n in self.params: 
            self.trainable_name += [n]
            p = self.params[n]
            self.trainable_parameters += [torch.rand_like(p.data, device=self.device, requires_grad=False)]
        
        self.num_params = sum([p.numel() for p in self.trainable_parameters])  

        self.vector = {}

        for key in self.trainable_name:
            # print(pretrained_state_dict[key].dtype)
            if key in self.pretrained_state_dict and key in self.finetuned_state_dict and self.pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                print(f"Key {key} has dtype {self.pretrained_state_dict[key].dtype} -- skipping!")
                continue
            self.vector[key] = (self.finetuned_state_dict[key] - self.pretrained_state_dict[key])


    def reset_model(self):
        with torch.no_grad():
            for name in self.trainable_name:
                if name in self.pretrained_state_dict and name in self.model_state_dict:
                    pretensor = self.pretrained_state_dict[name].to(self.device)
                    self.model_state_dict[name] += (pretensor - self.model_state_dict[name])

    

    def create_basepatch(self):
        """
        计算任务向量的绝对值。
        确定前 k 个最大值的阈值。
        创建一个与可训练参数形状相同的零张量列表。
        根据阈值设置掩码值。
        打印基础掩码中总参数的数量。
        返回基础掩码。
        
        """
        threshold = int(self.graft_args.sparsity * self.num_params)  

        abs_tv = []
        for key in self.task_vectors:
            abs_tv.append(torch.abs(self.task_vectors[key]).view(-1))
        # 把abs_tv展开为一个维度的
        abs_tv = torch.cat(abs_tv)
        k = int(self.graft_args.sparsity * abs_tv.numel())  # 1% of the total number of elements

        # Get the k largest values; returns values and their indices
        values, indices = torch.topk(abs_tv.view(-1), k)
        threshold = values.min()

        basepatch = {key: torch.zeros_like(self.task_vectors[key], requires_grad=False) for key in self.task_vectors}

        for key in self.task_vectors:
            p = self.task_vectors[key]
            q = basepatch[key]
            q[torch.absolute(p) > threshold] = self.graft_args.sigmoid_bias
            q[torch.absolute(p) <= threshold] = -self.graft_args.sigmoid_bias

        total_params = sum([torch.sum(torch.round(torch.nn.Sigmoid()(p)) * torch.round(torch.nn.Sigmoid()(p))) / (1. * self.num_params) for p in basepatch.values()])
        print('Total parameters in my stitch: ', total_params)
            
        return basepatch
    
    def interpolate_model(self, round_, return_mask=False):
        sigmoid = torch.nn.Sigmoid()

        n_graft_params, n_total_params = 0, 0

        binary_mask = {}
        with torch.no_grad():
            for key in self.trainable_name:
                if key in self.pretrained_state_dict and key in self.finetuned_state_dict and key in self.model_state_dict:
                    pretensor = self.pretrained_state_dict[key].to(self.device)
                    finetensor = self.finetuned_state_dict[key].to(self.device)
                    p = self.model_state_dict[key]

                    frac = sigmoid(self.mask[key])
                    if round_:
                        frac = torch.round(frac)
                        binary_mask[key] = frac
                    n_graft_params += torch.sum(frac)
                    frac = frac.to(self.device)
                    p += frac * (finetensor - pretensor)

        if round_:
            print('Proportion in my graft: ', n_graft_params / self.num_params)

        if return_mask:
            return binary_mask, n_graft_params / self.num_params
        
    def evaluate_vision(self, dataloader, dataset_name):
        classification_head = get_classification_head(self.args, dataset_name)
        model = ImageClassifier(self.model, classification_head)

        model.eval()

        with torch.no_grad():
            top1, correct, n = 0., 0., 0.
            for i, data in enumerate(tqdm(dataloader)):
                data = maybe_dictionarize(data)
                x = data['images'].to(self.device)
                y = data['labels'].to(self.device)

                logits = get_logits(x, model)

                pred = logits.argmax(dim=1, keepdim=True).to(self.device)

                correct += pred.eq(y.view_as(pred)).sum().item()
                
                n += y.size(0)

            top1 = correct / n

        metrics = {'top1': top1}
        print(f'Grafting on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
        return metrics

    def compress_task_vector(self):
        """
        Compress the task vector using the mask.

        Args:
            task_vector (dict of torch.Tensor): The task vector to be compressed.
            mask (dict of torch.Tensor): The mask indicating which elements to keep.

        Returns:
            sparse_vector (dict): The compressed task vector with indices and values.
        """
        assert self.task_vector.keys() == self.mask.keys(), "Task vector and mask must have the same keys"

        sparse_vector = {}
        for key in self.task_vector:

            tv = self.task_vector[key]
            m = self.mask[key]
            assert tv.shape == m.shape, "Each tensor in task vector and mask must have the same shape"
            print(tv.shape,m.shape)
            idx = torch.nonzero(m, as_tuple=True)
            if idx[0].numel() == 0:
                continue  # 如果索引为空，跳过该张量
            idx = [i.to(tv.device) for i in idx]
            val = tv[idx]
            sparse_vector[key] = {'indices': idx, 'values': val}

        return sparse_vector

    def train_graft(self, dataloader, dataset_name):
        loss_fct = torch.nn.CrossEntropyLoss()
        sigmoid = torch.nn.Sigmoid()
        
        device = self.device
        lr = self.graft_args.learning_rate
        
        for epoch in tqdm(range(self.graft_args.num_train_epochs), 'Training the mask'):
            print("Epoch: ", epoch)
            total_grad = {}
            
            self.interpolate_model(round_=True)

            for data in dataloader:
                if self.model_type == 'vit':
                    data = maybe_dictionarize(data)
                    x = data['images'].to(self.device)
                    y = data['labels'].to(self.device)
                    features = self.model(x)
                    outputs = self.classifier_head(features)
                    loss = loss_fct(outputs, y)

                loss.backward()
                
                null_grad_trainable_name = []
                for n, p in self.model.named_parameters():
                    if n in self.trainable_name and p.grad is None:
                        null_grad_trainable_name.append(n)

                grad = {}
                for n, p in self.model.named_parameters():
                    if n in self.trainable_name and n not in null_grad_trainable_name:
                        grad[n] = p.grad.detach().clone()
                    elif n in self.trainable_name and n in null_grad_trainable_name:
                        grad[n] = torch.zeros_like(p).detach().clone()
                self.model.zero_grad()
                for n in grad:
                    grad[n] = grad[n] * self.task_vectors[n].to(device)
                    if n not in total_grad:
                        total_grad[n] = lr * grad[n]
                    else:
                        total_grad[n] += lr * grad[n]

            for n in total_grad:
                total_grad[n] /= len(dataloader)
            
            self.reset_model()
    
            # Take the gradient step
            with torch.no_grad():
                for n in self.mask:
                    p = self.mask[n]
                    g = total_grad[n]
                    derivative = sigmoid(p) * (1 - sigmoid(p))
                    reg_term = self.graft_args.l1_strength * torch.where(p > 0, derivative, -derivative)
                    p -= g * derivative - reg_term

            ######## Evaluation of current mask ###########
            if (epoch + 1) % 5 == 0 or epoch == self.graft_args.num_train_epochs - 1:
                mask, proportion = self.interpolate_model(round_=False, return_mask=True)
                self.reset_model()

        mask, proportion = self.interpolate_model(round_=True, return_mask=True)
        self.reset_model()
        # 根据mask对task vector进行操作，保留我需要的值
        sparse_vector = self.compress_task_vector()

        return sparse_vector
    
class Stitcher(nn.Module):
    def __init__(self, trainable_params, model, pretrained_model, finetuned_models, later_tv):
        super(Stitcher, self).__init__()
        self.params = trainable_params
        self.pretrained_model = pretrained_model
        self.finetuned_models = finetuned_models
        self.model = model
        self.client_sparse_tvs = []
        self.merged_tv = later_tv
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    def merge_sparse_tvs(self):
        """
        合并多个稀疏任务向量，重叠索引处保留绝对值最大的值（支持任意维度索引）
        
        Args:
            sparse_tvs: 待合并的稀疏任务向量列表，每个元素为字典，结构为：
                    {
                        "layer_name1": {
                            "indices": (tensor_dim1, tensor_dim2, ...),  # 各维度索引张量的元组
                            "values": tensor_values                      # 值张量
                        },
                        ...
                    }
        
        Returns:
            合并后的稀疏任务向量，结构与输入相同
        """
        if self.merged_tv is not None:
            self.client_sparse_tvs.append(self.merged_tv)
        merged_tv = {}
        
        for sparse_tv in self.client_sparse_tvs:  # 遍历每个任务向量
            for key, data in sparse_tv.items():  # 遍历每层的稀疏数据
                new_indices = data['indices']
                new_values = data['values']
                
                # 初始化：如果该层尚未在合并结果中，直接添加
                if key not in merged_tv:
                    merged_tv[key] = {
                        'indices': new_indices,
                        'values': new_values
                    }
                    continue
                    
                # 获取该层现有的索引和值
                existing_data = merged_tv[key]
                existing_indices = existing_data['indices']
                existing_values = existing_data['values']
                
                # 空值检查
                if existing_values.numel() == 0:
                    merged_tv[key] = {'indices': new_indices, 'values': new_values}
                    continue
                if new_values.numel() == 0:
                    continue
                    
                # Step 1: 将现有索引转换为哈希表 {索引元组: 位置}
                index_map = {}
                for j in range(len(existing_values)):
                    # 提取第j个索引元组（适用于任意维度）
                    idx_tuple = tuple(idx[j].item() for idx in existing_indices)
                    index_map[idx_tuple] = j  # 记录位置
                    
                # Step 2: 收集需要新增的索引和值
                new_indices_to_add = [[] for _ in range(len(existing_indices))]  # 按维度存储
                new_values_to_add = []
                
                # 遍历新索引的每个元素
                for i in range(len(new_values)):
                    # 提取当前索引元组（如 (2,3) 或 (1,4,5)）
                    current_idx = tuple(idx[i].item() for idx in new_indices)
                    
                    # 检查是否已存在
                    if current_idx in index_map:
                        # 存在：比较绝对值，保留较大的值
                        existing_pos = index_map[current_idx]
                        if abs(new_values[i]) > abs(existing_values[existing_pos]):
                            existing_values[existing_pos] = new_values[i]
                    else:
                        # 不存在：记录需要新增的数据
                        for dim in range(len(new_indices)):
                            new_indices_to_add[dim].append(new_indices[dim][i].item())
                        new_values_to_add.append(new_values[i].item())
                        # 更新哈希表（位置为原长度 + 新增后的偏移量）
                        index_map[current_idx] = len(existing_values) + len(new_values_to_add) - 1
                        
                # Step 3: 批量拼接新增数据（减少张量操作次数）
                if new_values_to_add:
                    # 转换新增索引为张量（保持与原数据相同的设备和类型）
                    device = existing_indices[0].device
                    added_indices = [
                        torch.tensor(dim_indices, dtype=torch.long, device=device)
                        for dim_indices in new_indices_to_add
                    ]
                    # 拼接各维度索引
                    merged_indices = [
                        torch.cat((existing_indices[dim], added_indices[dim]))
                        for dim in range(len(existing_indices))
                    ]
                    # 拼接值张量
                    merged_values = torch.cat((existing_values, torch.tensor(new_values_to_add, device=device)))
                    
                    # 更新合并结果
                    merged_tv[key] = {'indices': merged_indices, 'values': merged_values}
        self.merged_tv = merged_tv
        return merged_tv
    
    
def apply_merged_tv_to_model(self):
    """
    将合并后的稀疏任务向量加到预训练模型上

    Args:
        model: 预训练模型
        merged_tv: 合并后的稀疏任务向量
    """
    model_state_dict = self.pretrained_model.state_dict()
    for key, data in self.merged_tv.items():
        indices = data['indices']
        values = data['values']
        # 创建全零张量（与模型参数同形状、同设备）
        dense_tensor = torch.zeros_like(model_state_dict[key], device=model_state_dict[key].device)
        dense_tensor[tuple(indices)] = values
        # 将任务向量加到模型参数上
        model_state_dict[key].add_(dense_tensor)
    # 加载更新后的参数
    self.pretrained_model.load_state_dict(model_state_dict)
