from server_imagenetR import vit_sparse_tv_imagenetR
from src.args import parse_arguments
from src.datasets.registry import get_dataset, registry
from src.modeling import ImageEncoder, ImageClassifier


if __name__ == '__main__':
    args = parse_arguments()
    args.model = 'ViT-B-16'
    args.dataset = 'ImageNetR'

    args.select_client = 2
    args.num_users = 30
    args.epochs = 1
    args.batch_size = 128

    args.task_num = 10

    args.num_train_epochs = 1
    args.sparsity = 1e-1

    args.federated_continual = True
    federated_continual_ft_dir = 'federated_continual/' if args.federated_continual else ''
    args.save = f'checkpoints/{args.model}/{federated_continual_ft_dir}domain_incremental'
    dataset_class = registry[args.dataset]
    method = 'seq-ft' if args.federated_continual else 'ind-ft'
    name = f'ft-{args.dataset}-DIL-{method}'

    #初始化模型
    initital_image_encoder = ImageEncoder(args,keep_lang=True)
    initital_image_encoder_dic = initital_image_encoder.state_dict()
    trainable_params = {}
    frozen = ["model.positional_embedding", "model.text_projection", "model.logit_scale",
            "model.token_embedding.weight", "model.ln_final.weight", "model.ln_final.bias"]
    for name, param in initital_image_encoder.named_parameters():
        if name.startswith("model.transformer") and name not in frozen:
            frozen.append(name)
    for k, v in initital_image_encoder_dic.items():
        if k not in frozen:
            trainable_params[k] = v

    global_model = vit_sparse_tv_imagenetR(args,initital_image_encoder,args.dataset,args.device,args.task_num,args.select_client,args.num_users,trainable_params,frozen)
    for current_task,domain_idx in enumerate(dataset_class.default_domain_order):

        args.subset_config = {
            'domains': [dataset_class.BASE_CLASS.DOMAINS[domain_idx]],
            'classes': dataset_class.BASE_CLASS.CLASSES,
        }

        print('='*100)
        print(f'Finetuning {args.model} on {args.dataset} & {current_task}')
        print('='*100)
        global_model.setup_data(args.subset_config)
        Flag = global_model.setup_task_data_and_classification_head(current_task)
        if Flag:
            print(f'task {current_task} Has Been Trained,Continual')
            continue
        global_model.train(current_task)
        global_model.afterTrain(current_task)


