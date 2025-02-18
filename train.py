import torch
from src.args import parse_arguments
from src.modeling import ImageEncoder
from server import vit_sparse_tv


if __name__ == '__main__':
    args = parse_arguments()
    args.model = 'ViT-B-16'
    args.n_splits = 10
    args.split_strategy = 'class'
    args.dataset = 'ImageNetR'
    args.select_client = 10
    args.num_users = 30
    # args.eval_datasets = 'CIFAR100'
    args.lr = 1e-5
    args.epochs = 5
    args.batch_size = 128
    args.task_num = 10
    args.num_train_epochs = 5
    args.sparsity = 1e-1
    args.federated_continual = True
    federated_continual_ft_dir = 'federated_continual/' if args.federated_continual else ''
    args.save = f'checkpoints/{args.model}/{federated_continual_ft_dir}{args.split_strategy}_incremental'
    args.model = 'ViT-B-16'

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
    # initital_image_encoder_ft = r'C:\vit_sparse_tv\checkpoints\ViT-B-16\federated_continual\class_incremental\CIFAR100-10\ft-epochs-5-seed_5-client_10\image_encoder_9.pt'
    # initital_image_encoder = torch.load(initital_image_encoder_ft)
    global_model = vit_sparse_tv(args,initital_image_encoder,args.dataset,args.device,args.task_num,args.select_client,args.num_users,trainable_params,frozen)
    global_model.setup_data(n_shot=args.n_shot)
    for current_task in range(args.n_splits):
        print(f"\n##### SPLIT {current_task} #####")
        Flag = global_model.setup_task_data_and_classification_head(current_task)
        if Flag:
            print(f'task {current_task} Has Been Trained,Continual')
            continue
        global_model.train(current_task)
        global_model.afterTrain(current_task)