# -*- coding: utf-8 -*-
import argparse
import torch
import time
from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")


def get_config():
    global current_time
    """使用argparse获取配置参数，支持命令行传入"""
    parser = argparse.ArgumentParser(description='Transformer_stock_predict')

    # 数据相关配置
    data_group = parser.add_argument_group('data_config')
    data_group.add_argument('--data_path', type=str, default='./data/min_eth_data.csv', help='the path of data')
    data_group.add_argument('--sequence_length', type=int, default=100, help='the length of input sequence')
    data_group.add_argument('--prediction_length', type=int, default=1, help='the length of prediction sequence')
    data_group.add_argument('--test_size', type=float, default=0.2, help='the size of test data')
    data_group.add_argument('--val_size', type=float, default=0.1, help='the size of validation data')
    data_group.add_argument('--batch_size', type=int, default=128, help='the batch size')
    data_group.add_argument('--shuffle', action='store_true', default=True, help='whether to shuffle data')
    data_group.add_argument('--normalization', type=str, default='minmax', choices=['minmax', 'standard', 'none'],
                            help='the normalization method')

    # 模型相关配置
    model_group = parser.add_argument_group('model_config')
    model_group.add_argument('--model_type', type=str, default='encoder', choices=['encoder', 'decoder'],
                             help="encoder or decoder")
    model_group.add_argument('--max_len', type=int, default=520, help='the maximum length of input sequence')
    model_group.add_argument('--input_dim', type=int, default=6, help='the dimension of input sequence')
    model_group.add_argument('--output_dim', type=int, default=1, help='the dimension of output')
    model_group.add_argument('--d_model', type=int, default=512, help='the dimension of models')
    model_group.add_argument('--n_head', type=int, default=8, help='the number of attention heads')
    model_group.add_argument('--num_layers', type=int, default=3, help='the number of transformer layers')
    model_group.add_argument('--dim_feedforward', type=int, default=2048, help='the dimension of ff') #4 * d_model
    model_group.add_argument('--dropout', type=float, default=0.1, help='the dropout ratio')
    model_group.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu'],
                             help='the style of activation function')

    # 训练相关配置
    train_group = parser.add_argument_group('train_config')
    train_group.add_argument('--epochs', type=int, default=30,
                             help='epochs')
    train_group.add_argument('--lr', type=float, default=1e-3,
                             help='initial learning rate')
    train_group.add_argument('--weight_decay', type=float, default=1e-5,
                             help='weight decay')
    train_group.add_argument('--loss_fn', type=str, default='huber',
                             choices=['mse','huber', 'mae', 'huber'],
                             help='loss function') # HuberLoss（鲁棒损失），对异常值的敏感度低
    train_group.add_argument('--optimizer', type=str, default='adam',
                             choices=['adam', 'sgd', 'rmsprop'],
                             help='optimizer style')
    train_group.add_argument('--lr_scheduler', type=str, default='cosine',
                             choices=['cosine', 'step', 'none'],
                             help='lr_scheduler style')
    train_group.add_argument('--is_early_stopping', type=bool, default=True, help='whether to early stopping')
    train_group.add_argument('--early_stopping_patience', type=int, default=3,
                             help='early_stopping_patience')
    train_group.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                             choices=['cpu', 'cuda'],
                             help='training device')

    # 日志和保存配置
    log_group = parser.add_argument_group('日志和保存配置')
    log_group.add_argument('--log_level', type=str, default='INFO',
                           choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                           help='the logging level')

    log_group.add_argument('--log_file', type=str, default=f'./log/logs/{current_time}-training.log',
                           help='the path of log file')
    log_group.add_argument('--log_format', type=str, default='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           help='the format of log file')
    log_group.add_argument('--save_dir', type=str, default='./log/models/',
                           help='the path of saved models')
    log_group.add_argument('--save_best', action='store_true', default=True,
                           help='whether to save best models')
    log_group.add_argument('--plot_dir', type=str, default='./log/plots/',
                           help='the path of plot file')

    # 解析参数
    args = parser.parse_args()

    # 转换为字典，便于访问
    config = vars(args)

    # 添加一些自动计算的配置
    config['device'] = torch.device(config['device'])

    return config


# 创建全局配置实例
config = get_config()
