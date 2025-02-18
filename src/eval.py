import os
import json
import tqdm
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
from src import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from src.datasets.registry import get_dataset
from src.cl_utils import get_dataset_and_classifier_for_split



def eval_given_dataset(image_encoder, dataset, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name, classnames=dataset.classnames)
    model = ImageClassifier(image_encoder, classification_head)
    dataloader = dataset.test_loader
    metrics = do_eval(model, dataloader, args.device)
    
    print(f"Done evaluating. Accuracy: {metrics['top1']:.4f}")
    
    return metrics


def eval_single_dataset(image_encoder, dataset_name, args):
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=None)

    metrics = do_eval(model, dataloader, args.device)
    
    print(f"Done evaluating on {dataset_name}. Accuracy: {metrics['top1']:.4f}")
    
    return metrics


def eval_task_aware(image_encoder, args):
    text_encoder = ImageEncoder(args, keep_lang=True)
    full_dataset = get_dataset(
        args.dataset,
        image_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )

    accs = []
    for split_idx in range(args.n_splits):
        dataset = deepcopy(full_dataset)
        dataset, classification_head,classes = get_dataset_and_classifier_for_split(
            dataset, split_idx, text_encoder, args 
        )
        model = ImageClassifier(image_encoder, classification_head)
        metrics = do_eval(model, dataset.test_loader, args.device)
        accs.append(metrics['top1'])
        print(f"Task-aware eval on split {split_idx} of dataset {args.dataset}. Accuracy: {accs[-1]}")
        
    return accs


def eval_task_agnostic(image_encoder, args,current_id):
    #这里是获取分类头
    classification_head = get_classification_head(args, args.dataset,image_encoder)
    model = ImageClassifier(image_encoder, classification_head)

    full_dataset = get_dataset(
        args.dataset,
        image_encoder.train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )

    accs = []
    #原本是for split_idx in range(args.split_id):
    for split_idx in range(current_id+1):
        dataset = deepcopy(full_dataset)
        dataset = get_dataset_and_classifier_for_split(
            dataset, split_idx, None, args, remap_labels=False, return_classifier=False
        )
        metrics = do_eval(model, dataset.test_loader, args.device)
        accs.append(metrics['top1'])
        print(f"Task-agnostic eval on split {split_idx} of dataset {args.dataset}. Accuracy: {accs[-1]}")
        
    return accs


@torch.no_grad()
def do_eval(model, dl, device):    
    correct, n = 0., 0.
    model.eval()
    for data in tqdm.tqdm(dl):
        data = maybe_dictionarize(data)
        x = data['images'].to(device)
        y = data['labels'].to(device)

        logits = utils.get_logits(x, model)
        pred = logits.argmax(dim=1, keepdim=True).to(device)
        correct += pred.eq(y.view_as(pred)).sum().item()
        n += y.size(0)

    metrics = {'top1': correct / n}
    
    return metrics


def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for dataset_name in args.eval_datasets:
        print('Evaluating on', dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ':' + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, 'a+') as f:
            f.write(json.dumps(info) + '\n')
        print(f'Results saved to {args.results_db}.')
    else:
        print('Results not saved (to do so, use --results_db to specify a path).')

    return info

"""
对过往类进行判断
"""
def per_cls_acc(self, val_loader, model):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (_, input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()
            # compute output
            output = model(input)["logits"]
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cf = confusion_matrix(all_targets, all_preds).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt
    return cls_acc
    # pdb.set_trace()
    # out_cls_acc = 'Per Class Accuracy: %s' % ((np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
    # print(out_cls_acc)

