# -*- coding: utf-8 -*-
import os
import time
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import config
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger("trainer")


class Trainer:
    """模型训练器类，负责模型的训练、验证和评估"""

    def __init__(self, model, train_loader, val_loader, test_loader, data_processor):
        self.model = model.to(config['device'])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.data_processor = data_processor

        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_lr_scheduler()
        self.criterion = self._get_loss_function()

        # 训练过程记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

        # 创建保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(config['plot_dir'], exist_ok=True)

    def _get_optimizer(self):
        """根据配置获取优化器"""
        optimizer_name = config['optimizer'].lower()
        lr = config['lr']
        weight_decay = config['weight_decay']

        logger.info(f"配置优化器{lr},权重衰减为{weight_decay}")
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

    def _get_lr_scheduler(self):
        """根据配置获取学习率调度器"""
        scheduler_name = config['lr_scheduler'].lower()

        logger.info(f"配置学习率调度器{scheduler_name}")
        if scheduler_name == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        elif scheduler_name == 'step':
            return StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"不支持的学习率调度器: {scheduler_name}")

    def _get_loss_function(self):
        """根据配置获取损失函数"""
        loss_name = config['loss_fn'].lower()

        logger.info(f"配置损失函数{loss_name}")
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'huber':
            return nn.HuberLoss(delta=1.0)
        elif loss_name == 'mae':
            return nn.L1Loss()
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")

    def train_one_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (inputs, targets) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            # 移动数据到设备
            inputs = inputs.to(config['device'])
            targets = targets.to(config['device'])

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            if config['model_type'] == 'decoder':
                # 取目标的前n-1步作为解码器输入
                decoder_input = torch.cat([torch.zeros_like(targets[:, :1, :]), targets[:, :-1, :]], dim=1)
                outputs = self.model(inputs, decoder_input).squeeze(-1)
            else:
                outputs = self.model(inputs).squeeze(-1) # (batch_size,seq_len,input_dim)

            # 计算损失
            loss = self.criterion(outputs, targets)

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 打印批次信息
            if batch_idx % 10 == 0:
                logger.debug(
                    f"Epoch {epoch + 1}/{config['epochs']}, Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item():.6f}")

        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # 移动数据到设备
                inputs = inputs.to(config['device'])
                targets = targets.to(config['device'])

                # 前向传播
                if config['model_type'] == 'decoder':
                    decoder_input = torch.cat([torch.zeros_like(targets[:, :1, :]), targets[:, :-1, :]], dim=1)
                    outputs = self.model(inputs, decoder_input).squeeze(-1)
                else:
                    outputs = self.model(inputs).squeeze(-1)

                # 计算损失
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # 保存结果用于后续分析
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # 计算平均损失
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)

        return avg_loss

    def train(self):
        """完整训练过程"""
        logger.info("开始模型训练...")
        start_time = time.time()

        for epoch in range(config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{config['epochs']}")
            # 训练一个epoch
            train_loss = self.train_one_epoch(epoch)

            # 验证
            val_loss = self.validate()

            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()

            # 打印 epoch 信息
            logger.info(
                f"Epoch {epoch + 1}/{config['epochs']} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"LR: {self.optimizer.param_groups[0]['lr']:.8f}"
            )

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(f"best_model.pt")
                self.early_stopping_counter = 0
                # 重置学习率调度器
                if self.scheduler is not None and config.get('reset_scheduler_on_best', False):
                    self.scheduler = self._get_lr_scheduler()
                logger.info(f"保存最佳模型，验证损失: {val_loss:.6f}")
            else:
                self.early_stopping_counter += 1
                logger.info(f"早停计数器: {self.early_stopping_counter}/{config['early_stopping_patience']}")

                # 早停机制
                if config['is_early_stopping']:
                    if self.early_stopping_counter >= config['early_stopping_patience']:
                        logger.info(f"触发早停机制，停止训练")
                        break

        # 训练结束后保存最后一个模型（如果配置了）
        if not config['save_best']:
            self.save_model(f"final_model.pt")

        # 绘制损失曲线
        self.plot_loss_curve()

        end_time = time.time()
        logger.info(f"训练完成 - 耗时: {((end_time - start_time) / 60):.2f} 分钟")

        return self

    def evaluate(self):
        """评估模型在测试集上的性能"""
        logger.info("开始模型评估...")

        # 加载最佳模型
        self.load_model("best_model.pt")
        self.model.eval()

        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                # 移动数据到设备
                inputs = inputs.to(config['device'])
                targets = targets.to(config['device'])

                # 前向传播
                if config['model_type'] == 'decoder':
                    # 推理时自回归生成
                    batch_size = inputs.size(0)
                    pred_len = config['prediction_length']
                    output_dim = config['output_dim']

                    # 初始化解码器输入
                    decoder_input = torch.zeros(batch_size, 1, output_dim).to(config['device'])
                    outputs = []

                    # 自回归生成
                    for _ in range(pred_len):
                        step_output = self.model(inputs, decoder_input)
                        last_output = step_output[:, -1:, :]  # 取最后一步的输出
                        outputs.append(last_output)

                        # 将输出作为下一步的输入
                        decoder_input = torch.cat([decoder_input, last_output], dim=1)

                    # 合并所有时间步的输出
                    outputs = torch.cat(outputs, dim=1)
                else:
                    outputs = self.model(inputs)

                # 保存结果
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # 合并所有批次的结果
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # 将标准化的数据转换回原始尺度
        all_outputs_original = self.data_processor.inverse_transform_target(all_outputs.reshape(-1))
        all_targets_original = self.data_processor.inverse_transform_target(all_targets.reshape(-1))

        # 计算整体评估指标
        metrics = self._calculate_metrics(all_targets_original, all_outputs_original)

        # 按预测步长计算指标
        step_metrics = self._calculate_step_metrics(all_targets, all_outputs)
        metrics['step_metrics'] = step_metrics

        # 绘制预测结果
        self.plot_predictions(all_outputs_original, all_targets_original)

        return {
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'r2': metrics['r2'],
            'predictions': all_outputs_original,
            'targets': all_targets_original
        }

    def _calculate_metrics(self, targets, predictions):
        """计算评估指标"""
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)

        logger.info(f"整体评估指标 - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def _calculate_step_metrics(self, targets, predictions):
        """按预测步长计算评估指标"""
        pred_len = config['prediction_length']
        step_metrics = []
        # 正确逻辑：先展平为1维，反标准化后再按步长拆分
        # 1. 展平目标值和预测值（样本数×pred_len → 总长度=样本数×pred_len 的1D数组）
        targets_flat = targets.reshape(-1)  # 形状：(total_samples,)
        predictions_flat = predictions.reshape(-1)  # 形状：(total_samples,)
        # 2. 反标准化（符合 inverse_transform_target 的输入要求）
        targets_original = self.data_processor.inverse_transform_target(targets_flat)
        predictions_original = self.data_processor.inverse_transform_target(predictions_flat)
        # 3. 按预测步长拆分（恢复为“样本数×pred_len”的2D数组）
        targets_original = targets_original.reshape(-1, pred_len)  # 形状：(num_samples, pred_len)
        predictions_original = predictions_original.reshape(-1, pred_len)  # 形状：(num_samples, pred_len)
        # 4. 逐步计算指标
        for step in range(pred_len):
            step_targets = targets_original[:, step]  # 第step步的所有真实值
            step_preds = predictions_original[:, step]  # 第step步的所有预测值
            mae = mean_absolute_error(step_targets, step_preds)
            rmse = np.sqrt(mean_squared_error(step_targets, step_preds))
            r2 = r2_score(step_targets, step_preds)
            step_metrics.append({'step': step + 1, 'mae': mae, 'rmse': rmse, 'r2': r2})
            logger.info(f"步长 {step + 1} 指标 - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

        return step_metrics

    def save_model(self, filename):
        """保存模型"""
        model_path = os.path.join(config['save_dir'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': config
        }, model_path)
        logger.info(f"模型已保存到 {model_path}")

    def load_model(self, filename):
        """加载模型"""
        model_path = os.path.join(config['save_dir'], filename)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=config['device'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_val_loss = checkpoint['best_val_loss']
            logger.info(f"已加载模型: {model_path}")
        else:
            logger.warning(f"模型文件不存在: {model_path}")

    def plot_loss_curve(self):
        """绘制训练和验证损失曲线"""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Train Loss')
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Curve')
            plt.legend()
            plt.grid(True)

            # 保存图像
            plot_path = os.path.join(config['plot_dir'], 'loss_curve.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"损失曲线已保存到 {plot_path}")
            plt.close()
        except Exception as e:
            logger.error(f"绘制损失曲线失败: {e}")

    def plot_predictions(self, predictions, targets):
        """绘制预测结果与真实值对比图"""
        try:
            plt.figure(figsize=(15, 8))

            # 绘制全部预测结果
            plt.subplot(2, 1, 1)
            plt.plot(targets, label='True Price', alpha=0.7)
            plt.plot(predictions, label='Predicted Price', alpha=0.7)
            plt.xlabel('Time Step')
            plt.ylabel('Price')
            plt.title('Price Prediction vs True Price')
            plt.legend()
            plt.grid(True)

            # 绘制部分预测结果（最后100个时间步）
            plt.subplot(2, 1, 2)
            start_idx = max(0, len(targets) - 100)
            plt.plot(range(start_idx, len(targets)), targets[start_idx:], label='True Price', alpha=0.7)
            plt.plot(range(start_idx, len(predictions)), predictions[start_idx:], label='Predicted Price', alpha=0.7)
            plt.xlabel('Time Step')
            plt.ylabel('Price')
            plt.title('Price Prediction vs True Price (Last 100 Steps)')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()

            # 保存图像
            plot_path = os.path.join(config['plot_dir'], 'prediction_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"预测对比图已保存到 {plot_path}")
            plt.close()
        except Exception as e:
            logger.error(f"绘制预测对比图失败: {e}")
