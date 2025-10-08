# -*- coding: utf-8 -*-
import logging
import os


def setup_logger(config):
    """配置日志系统"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志级别

    # 格式器：定义日志输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 文件处理器：将日志写入文件
    # 提取目录部分（不含文件名）
    log_dir = os.path.dirname(config['log_file'])
    # 递归创建目录（exist_ok=True 表示如果目录已存在也不报错）
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(config['log_file'], encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台处理器：同时在控制台输出日志
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
