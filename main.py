# -*- coding: utf-8 -*-
import logging
import os
from data.data_loader import DataProcessor
from models.transformer import get_transformer_model
from trainer.trainer import Trainer
from log.logs.logger import *
from config import config


def main():
    """主函数，协调数据加载、模型初始化和训练过程"""
    # 配置日志
    logger = setup_logger(config)
    logger.info("===== 价格预测模型开始运行 =====")

    try:
        # 1. 数据处理
        logger.info("开始数据处理...")
        data_processor = DataProcessor()
        data_processor.load_data()
        data_processor.preprocess_data()
        train_loader, val_loader, test_loader = data_processor.get_dataloaders()
        logger.info(
            f"数据处理完成 - 训练集: {len(train_loader.dataset)}, 验证集: {len(val_loader.dataset)}, 测试集: {len(test_loader.dataset)}")

        # 2. 初始化模型
        logger.info("初始化模型...")
        model = get_transformer_model()
        logger.info(f"模型类型: {config['model_type']} 型Transformer")
        logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # 3. 初始化训练器并训练
        logger.info("开始模型训练...")
        trainer = Trainer(model, train_loader, val_loader, test_loader, data_processor)
        trainer.train()

        logger.info("===== 价格预测模型运行结束 =====")

        # 4. 评估模型
        # logger.info("开始模型评估...")
        evaluation_results = trainer.evaluate()

    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
