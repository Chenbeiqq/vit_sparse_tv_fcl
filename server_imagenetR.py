import copy
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from src.localize_utils import  Stitcher
from client import LocalUpdate
from src.cl_utils import get_dataset_and_classifier_for_split, create_non_iid_dataloaders,create_non_iid_dataloaders_with_val
from src.datasets.common import maybe_dictionarize, get_dataloader
from src.datasets.registry import get_dataset, create_k_shot_dataset
from src.heads import get_classification_head
from src.merging.task_vectors import merge_max_abs
from src.modeling import ImageClassifier, ImageEncoder
from src.utils import get_logits
import tqdm
class vit_sparse_tv_imagenetR:
    def __init__(self,args,image_encoder,dataset_name,device,task_num,client,
                 all_client,trainable_params,frozen):
        self.args = args
        self.dataset_name = dataset_name
        self.tv = None
        self.global_image_encoder = image_encoder
        self.classification_head = None
        self.global_model = None
        self.preprocess_fn = self.global_image_encoder.train_preprocess
        self.classes = None
        self.device = device
        self.dataset = None
        self.task_sub_dataset = None
        #task num denote a task include classes
        self.task_num = task_num
        self.epochs = args.epochs
        self.select_client = client
        self.all_client = all_client
        self.global_image_encoder_ft = os.path.join(self.args.save, f"{self.dataset_name}",
                                                    f"ft-epochs-{args.epochs}-graft_epoch-{args.num_train_epochs}-seed_{args.seed}-client_{self.select_client}")
        self.classnames = None
        self.pretrained_model_dict = {}
        self.client_sparse_tv = []
        self.trainable_params = trainable_params
        self.frozen = frozen

    def federated_class_continual_eval(self,idx):
        test_classification_head = get_classification_head(self.args,self.dataset_name,self.global_image_encoder,classnames=self.classnames)
        model = ImageClassifier(self.global_image_encoder,test_classification_head)

        accs = []
        for split_idx in range(idx + 1):
            dataset = copy.deepcopy(self.dataset)
            dataset = get_dataset_and_classifier_for_split(
                dataset, split_idx, None, self.args, remap_labels=False, return_classifier=False,classnames=self.classnames
            )
            metrics = self.do_eval(model,dataset.test_loader, self.device)
            accs.append(metrics['top1'])
            print(f"Task eval on split {split_idx} of dataset {self.dataset_name}. Accuracy: {accs[-1]}")

        return accs

    @torch.no_grad()
    def do_eval(self, model,dl, device):
        correct, n = 0., 0.
        model.eval()
        for data in tqdm.tqdm(dl):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        metrics = {'top1': correct / n}

        return metrics
    def save_checkpoint(self,current_task):
        image_encoder_ft = os.path.join(self.global_image_encoder_ft,f'image_encoder_{current_task}.pt')
        self.global_image_encoder.save(image_encoder_ft)

    def save_stats_to_excel(self,stats_dict,current_task,excel_path='sparse_merge_stats.xlsx'):
        """
        将合并统计信息保存到Excel文件，支持追加新数据

        Args:
            stats_dict: 统计信息字典
            excel_path: Excel文件路径
        """
        # 准备当前统计数据
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        rows = []

        for layer_name, stats in stats_dict.items():
            row = {
                '时间': current_time,
                '层名称': layer_name,
                '当前任务': current_task,
                '总索引数': stats['total_indices'],
                '唯一索引数': stats['unique_indices'],
                '重复索引数': stats['duplicate_indices'],
                '重复率': f"{stats['duplicate_ratio']:.2%}",
                '最大重复次数': stats['max_duplicates'],
                '最大重复索引数': stats['indices_with_max_duplicates']
            }
            rows.append(row)

        # 创建新的DataFrame
        new_df = pd.DataFrame(rows)

        try:
            # 如果文件存在，读取并追加
            if os.path.exists(excel_path):
                existing_df = pd.read_excel(excel_path)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                updated_df = new_df

            # 保存到Excel
            updated_df.to_excel(excel_path, index=False)
            print(f"统计数据已保存到 {excel_path}")

        except Exception as e:
            print(f"保存Excel时发生错误: {str(e)}")
            # 尝试使用备份文件名保存
            backup_path = f'sparse_merge_stats_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            new_df.to_excel(backup_path, index=False)
            print(f"已保存备份文件到 {backup_path}")
    def setup_data(self,subset_config):
        assert self.dataset_name is not None, "please provide a dataset for training"
        self.subset_config = subset_config
        self.dataset = get_dataset(
            self.dataset_name,
            self.preprocess_fn,
            location=self.args.data_location,
            batch_size=self.args.batch_size,
            subset_config=self.subset_config
        )
        self.classnames = self.dataset.classnames

    def setup_task_data_and_classification_head(self,current_task):
        #检查是否已存在过去训练的image_encoder,要做到sequential finetuning
        if os.path.exists(os.path.join(self.global_image_encoder_ft, f'image_encoder_{current_task}.pt')):
            print(f"Skipping finetuning on split {current_task}, "
                  f"ckpt already exists under {os.path.join(self.global_image_encoder_ft, f'image_encoder_{current_task}.pt')}")
            return True
        if current_task>0:
            prev_image_encoder_ft = os.path.join(self.global_image_encoder_ft, f'image_encoder_{current_task-1}.pt')
            print(f'Loading image encoder from prev task {prev_image_encoder_ft=}')
            self.global_image_encoder = torch.load(prev_image_encoder_ft)
            #denote  pass this task process
        # dataset = copy.deepcopy(self.dataset)
        #构建任务分类头
        self.classification_head = get_classification_head(
            self.args,self.dataset_name, self.global_image_encoder,classnames=self.classnames
        )
        self.global_model = ImageClassifier(self.global_image_encoder,self.classification_head)
        self.global_model.freeze_head()

        return False
    #
    # def beforeTrain(self,current_task):
    #     self.setup_task_data_and_classification_head(current_task)

    def train(self,current_task):
        train_data_loader = get_dataloader(self.dataset, is_train=True, args=self.args, image_encoder=None)
        user_groups_loader,user_groups_valloader = create_non_iid_dataloaders_with_val(train_data_loader,n_parties=self.select_client,beta=0.5)
        idxs_users = np.random.choice(range(self.args.num_users), self.select_client, replace=False)
        map_dict = {}
        map_dict_val = {}
        for (key,value),new_key in zip(user_groups_loader.items(),idxs_users):
            map_dict[new_key] = value
        for (key, value), new_key in zip(user_groups_valloader.items(), idxs_users):
            map_dict_val[new_key] = value
        user_groups_loader = map_dict
        user_groups_valloader = map_dict_val

        for idx in idxs_users:
            print(f'client {idx} in {self.dataset_name}-task{current_task}')
            client = LocalUpdate(self.args,user_groups_loader[idx],self.trainable_params,n_shot=self.args.n_shot,val_dataloader=user_groups_valloader[idx])
            client_model = copy.deepcopy(self.global_model)
            self.client_sparse_tv.append(client.local_train(client_model))

        #aggregation client task vector
        if current_task == 0:
            merged_client_sparse_tv = Stitcher(self.client_sparse_tv,self.trainable_params,self.args,None,self.frozen)
        else:
            merged_client_sparse_tv = Stitcher(self.client_sparse_tv,self.trainable_params,self.args,self.tv,self.frozen)
        self.tv = merged_client_sparse_tv.merge_dense_tvs()
        # self.save_stats_to_excel(status,current_task)
        # 打印每层的重复统计信息

        self.global_image_encoder = merged_client_sparse_tv.apply_dense_tv_to_model(self.tv,scaling_coef=0.5)

        # merged_tv = merge_max_abs(local_client_tv)
        # self.global_image_encoder = merged_tv.apply_to(self.args.pretrained_checkpoint, scaling_coef=0.5)
        # self.tv = merged_tv

        self.save_checkpoint(current_task)
    def afterTrain(self,current_task):
        print('#' * 100 + "\nPerforming old task evaluation of federated continual learning .")
        results = []
        print(f"\nEVAL: {self.dataset_name}-{self.args.n_splits} (federated {self.args.split_strategy} incremental) - split idx: {current_task}")
        res = self.federated_class_continual_eval(current_task)
        results.append(res)
        print(f" eval on {self.dataset_name} after task {current_task}. Accuracies:\n{res}")

        print(f" evaluation  final results:\n{results}\n" + '#' * 100 + '\n')