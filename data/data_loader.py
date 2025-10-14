# -*- coding: utf-8 -*-

# import
import os
import torch
import logging
import numpy as np
import pandas as pd

# from import
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# own
from config import config

logger = logging.getLogger("data_loader")


class PriceDataset(Dataset):
    """价格预测数据集类"""

    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


class DataProcessor:
    """数据处理类，负责数据加载、预处理和数据集创建"""

    def __init__(self):
        self.config = config
        self.data = None
        self.X = None
        self.y = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.scaler = None
        self.scaler_target = None

        # 创建必要的目录
        self._create_directories()

    def _create_directories(self):
        for dir_path in [self.config['save_dir'], self.config['plot_dir'], os.path.dirname(self.config['log_file'])]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"创建目录：{dir_path}")

    def load_data(self):
        """加载原始数据"""
        try:
            logger.info(f"从{self.config['data_path']}加载数据")
            data = pd.read_csv(self.config['data_path'], index_col='t')
            data['c_pred'] = data['c'].shift(-1)
            self.data = data.dropna()
            logger.info(f"数据加载完成,形状:{self.data.shape}")
        except Exception as e:
            logger.error(f"数据加载失败:{str(e)}", exc_info=True)
            raise e

    def preprocess_data(self):
        """预处理数据，标准化和创建序列"""
        if self.data is None:
            raise ValueError("请先调用load_data()加载数据")

        logger.info("开始数据预处理...")

        # 选择特征列（排除目标列）
        target_column = 'c_pred'  # 目标价格列
        feature_columns = [col for col in self.data.columns if col != target_column]  # 特征列

        # 先对数据进行一阶差分
        self._time_series_differencer()

        # 数据标准化
        self._normalize_data(feature_columns, target_column)

        # 创建序列数据
        self._create_sequences(feature_columns, target_column)
        logger.info("数据预处理完成")

    def _time_series_differencer(self):
        """对有明显时序特征的数据进行差分处理"""
        logger.info("对有明显时序特征的数据进行差分进行处理")
        # 存储差分所需的起点值（用于还原）
        self.first_values = {}

        # 记录需要差分的特征
        diff_features = ['o', 'h', 'l', 'c', 'c_pred','v','qv']

        # 对当前的first_values进行初始化
        for feature in diff_features:
            self.first_values[feature] = self.data[feature].iloc[0]

        # 复制数据以避免修改原始数据
        diff_df = self.data.copy()

        # 对指定特征进行差分处理
        diff_df[diff_features] = self.data[diff_features].diff()

        # 移除第一行（因为差分后第一行为NaN）
        diff_df = diff_df.iloc[1:].copy()

        self.data[diff_features] = diff_df[diff_features]

        self.data.dropna(inplace=True)

        logger.info("对有明显时序特征的数据进行差分处理完成")

    def _normalize_data(self, feature_columns, target_column):
        """标准化特征和目标变量"""
        logger.info(f"使用{self.config['normalization']}标准化所有特征和目标变量")

        if self.config['normalization'] == 'minmax':
            self.scaler_features = MinMaxScaler(feature_range=(0, 1))
            self.scaler_target = MinMaxScaler(feature_range=(0, 1))
        elif self.config['normalization'] == 'standard':
            self.scaler_features = StandardScaler()
            self.scaler_target = StandardScaler()
        elif self.config['normalization'] == 'robust':
            self.scaler_features = RobustScaler()
            self.scaler_target = RobustScaler()

        # 划分时序训练/验证/测试集的索引（按时间顺序，不打乱）
        total_len = len(self.data)
        test_len = int(total_len * self.config['test_size'])
        val_len = int(total_len * self.config['val_size'])
        train_len = total_len - val_len - test_len

        # 时序索引：train→val→test（早期→中期→晚期）
        train_idx = range(train_len)
        val_idx = range(train_len, train_len + val_len)
        test_idx = range(train_len + val_len, total_len)

        # 拟合训练数据的特征
        self.scaler_features.fit(self.data.iloc[train_idx][feature_columns])

        # 拟合训练数据的目标
        self.scaler_target.fit(self.data.iloc[train_idx][[target_column]])

        # 转换特征
        self.data.loc[:, feature_columns] = self.scaler_features.transform(self.data[feature_columns])

        # 转换目标
        self.data.loc[:, target_column] = self.scaler_target.transform(self.data[[target_column]])

        logger.info(f"训练集特征标准化完成")
        logger.info(f"验证集特征标准化完成")
        logger.info(f"测试集特征标准化完成")

        logger.info(f"训练集特征均值:{self.data.iloc[train_idx][feature_columns].mean().values}")
        logger.info(f"验证集特征均值:{self.data.iloc[val_idx][feature_columns].mean().values}")
        logger.info(f"测试集特征均值:{self.data.iloc[test_idx][feature_columns].mean().values}")

    def _create_sequences(self, feature_columns, target_column):
        """创建输入序列和目标序列"""
        seq_len = self.config['sequence_length']
        pred_len = self.config['prediction_length']
        logger.info(f"创建序列 - 输入长度: {seq_len}, 预测长度: {pred_len}")
        X, y = [], []
        # 输入：[i, i+seq_len-1]（长度为 seq_len 的历史数据）
        # 目标：[i+seq_len, i+seq_len+pred_len-1]（长度为 pred_len 的未来数据）
        for i in tqdm(range(len(self.data) - seq_len - pred_len + 1)):
            # 输入序列
            seq = self.data[feature_columns].iloc[i:i + seq_len].values
            X.append(seq)

            # 目标序列(未来pred_len个时间步的价格)
            target = self.data[target_column].iloc[i + seq_len:i + seq_len + pred_len].values
            y.append(target)

        self.X = np.array(X)  # X:(list_num,seq_len,feature dimension)
        self.y = np.array(y)  # Y:(list_num,predict_len,)

        logger.info(f"序列创建完成 - 特征形状: {self.X.shape}, 目标形状: {self.y.shape}")

    def get_dataloaders(self):
        """创建并返回训练、验证和测试数据加载器"""
        if self.X is None or self.y is None:
            raise ValueError("请先调用 preprocess_data() 处理数据")

        # 划分训练集和测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y,
            test_size=self.config['test_size'],
            shuffle=False  # 时间序列不打乱顺序
        )

        # 从训练集中划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.config['val_size'] / (1 - self.config['test_size']),
            shuffle=False
        )

        # 创建数据集
        train_dataset = PriceDataset(X_train, y_train)
        val_dataset = PriceDataset(X_val, y_val)
        test_dataset = PriceDataset(X_test, y_test)

        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=self.config['shuffle']
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        logger.info(
            f"数据加载器创建完成 - 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

        return self.train_loader, self.val_loader, self.test_loader

    def inverse_transform_target(self, scaled_data):
        """将标准化的目标数据转换回原始尺度"""
        if self.scaler_target and self.config['normalization'] != 'none':
            return self.scaler_target.inverse_transform(scaled_data.reshape(-1, 1)).flatten()
        return scaled_data


if __name__ == '__main__':
    dataprocessor = DataProcessor()
    dataprocessor.load_data()  # 加载数据
    dataprocessor.preprocess_data()  # 数据处理
    dataprocessor.get_dataloaders()
    dataprocessor.inverse_transform_target(dataprocessor.y)
