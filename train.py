import csv
import os
from curses import wrapper
from datetime import datetime

from server import vit_sparse_tv
from src.args import parse_arguments
from src.logging import get_logger, log_dir
from src.modeling import ImageEncoder

train_logger = get_logger(__name__, "debug")

if __name__ == '__main__':
    # os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    args = parse_arguments()
    args.model = 'ViT-B-16'
    args.n_splits = 10
    args.split_strategy = 'class'
    args.dataset = 'CIFAR100'
    args.select_client = 10
    args.num_users = 30
    # args.eval_datasets = 'CIFAR100'
    args.lr = 1e-5
    args.epochs = 1
    args.batch_size = 16
    args.accumulation_steps = 8  # 梯度累积
    args.task_num = 10
    args.num_train_epochs = 1
    args.sparsity = 1e-1
    args.federated_continual = True
    federated_continual_ft_dir = 'federated_continual/' if args.federated_continual else ''
    args.save = f'checkpoints/{args.model}/{federated_continual_ft_dir}{args.split_strategy}_incremental'
    args.model = 'ViT-B-16'

    # 初始化模型
    initital_image_encoder = ImageEncoder(args, keep_lang=True)
    # initital_image_encoder.save(args.pretrained_checkpoint)
    initital_image_encoder_dic = initital_image_encoder.state_dict()
    trainable_params = {}
    # 冻结的参数为语言相关的参数
    frozen = ["model.positional_embedding", "model.text_projection", "model.logit_scale",
              "model.token_embedding.weight", "model.ln_final.weight", "model.ln_final.bias"]
    for name, param in initital_image_encoder.named_parameters():
        if name.startswith("model.transformer") and name not in frozen:
            frozen.append(name)
    # 得到可训练的参数
    for k, v in initital_image_encoder_dic.items():
        if k not in frozen:
            trainable_params[k] = v
    # initital_image_encoder_ft = r'C:\vit_sparse_tv\checkpoints\ViT-B-16\federated_continual\class_incremental\CIFAR100-10\ft-epochs-5-seed_5-client_10\image_encoder_9.pt'
    # initital_image_encoder = torch.load(initital_image_encoder_ft)
    global_model = vit_sparse_tv(args, initital_image_encoder, args.dataset, args.device, args.task_num,
                                 args.select_client, args.num_users, trainable_params, frozen)
    global_model.setup_data(n_shot=args.n_shot)

    results = []
    average_results = []
    for current_task in range(args.n_splits):
        task_start_time = datetime.now()
        train_logger.info(f"##### SPLIT {current_task} #####")

        Flag = global_model.setup_task_data_and_classification_head(current_task)
        if Flag:
            train_logger.info(f'task {current_task} Has Been Trained,Continual')
            continue
        global_model.train(current_task)
        result = global_model.afterTrain(current_task)

        # 计算平均结果并保存
        average_result = sum(result) / len(result)
        average_results.append(average_result)
        results.append(result)

        task_end_time = datetime.now()
        train_logger.info(f"Task {current_task} finished in {task_end_time - task_start_time}")

    # 写入结果数据
    train_logger.info("Final average result:" + str(sum(average_results) / len(average_results)))
    results_file = os.path.join(log_dir, 'results.csv')
    with open(results_file, 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(results)
        writer.writerows(average_results)
