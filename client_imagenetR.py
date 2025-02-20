import copy
import time
import numpy as np
import torch.backends.cuda as cudnn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from src.datasets.common import maybe_dictionarize
from src.localize_utils import Localizer
from src.utils import cosine_lr, LabelSmoothing
from src.merging.task_vectors import TaskVector, merge_max_abs
import warnings

class LocalUpdate(object):
    def __init__(self, args, sub_trainloader,trainable_params,n_shot,val_dataloader):
        self.mask = None
        self.args = args
        self.trainloader = sub_trainloader
        self.trainable_params = trainable_params
        self.client_mask = None
        self.val_dataloader = val_dataloader
        self.n_shot = n_shot
        self.final_model = None
        self.finetuned_model = None
        self.localizer = None
        self.pretrained_model = torch.load(args.pretrained_checkpoint,weights_only=False)
        self.classifier_head = None
        self.sparse_tv = {}
    def local_train(self, client_model):
        print_every = 100
        devices = list(range(torch.cuda.device_count()))
        print('Using devices', devices)
        client_model = torch.nn.DataParallel(client_model, device_ids=devices)
        if self.args.lr > 0:
            loss_fn = LabelSmoothing(self.args.ls)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        params = [p for p in client_model.parameters() if p.requires_grad]
        #开始获取image_encoder中的trainable_params
        # client_model_image_encoder_dic = client_model.module.image_encoder.state_dict()
        # for k,v in client_model_image_encoder_dic.items():
        #     if v.requires_grad:
        #         self.trainable_params[k] = v
        optimizer = torch.optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.wd)
        num_batches = len(self.trainloader)
        scheduler = cosine_lr(optimizer, self.args.lr, self.args.warmup_length, self.args.epochs * num_batches)
        n_batch = len(self.trainloader)
        for epoch in range(self.args.epochs):
            client_model = client_model.cuda()
            client_model.train()

            for i, batch in enumerate(self.trainloader):

                start_time = time.time()

                step = i + epoch * num_batches
                scheduler(step)
                optimizer.zero_grad()

                batch = maybe_dictionarize(batch)
                inputs = batch['images'].to('cuda:0')
                labels = batch['labels'].to('cuda:0')
                data_time = time.time() - start_time

                logits = client_model(inputs)

                loss = loss_fn(logits, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                optimizer.step()
                batch_time = time.time() - start_time

                if step % print_every == 0 or i + 1 == n_batch:
                    percent_complete = 100 * i / len(self.trainloader)
                    print(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(self.trainloader)}]\t"
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                    )

        # 开始生成task vector
        client_image_encoder = client_model.module.image_encoder
        classifier_head = client_model.module.classification_head
        self.finetuned_model = client_image_encoder
        self.classifier_head = classifier_head

        # client_task_vector = TaskVector(pretrained_checkpoint=self.args.pretrained_checkpoint,
        #                                 finetuned_state_dict=client_image_encoder.state_dict())
        self.afterTrain()
        return self.sparse_tv

    def afterTrain(self):
        self.final_model = copy.deepcopy(self.finetuned_model)
        self.localizer = Localizer(self.trainable_params,self.final_model,self.pretrained_model,self.finetuned_model, self.args.dataset,self.args,self.args,self.classifier_head,model_type='vit')
        self.sparse_tv = self.localizer.train_graft(self.val_dataloader,self.args.dataset)



