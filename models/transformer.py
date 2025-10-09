# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import config


class PositionalEncoding(nn.Module):
    """位置编码模块，为输入序列添加位置信息"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)  # 无须更新此参数

    def forward(self, x):
        """
        :param x:输入张量，形状为(seq_len, batch_size,d_model)
        :return:
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoderModel(nn.Module):
    """仅使用Transformer编码器的价格预测模型"""

    def __init__(self):
        super(TransformerEncoderModel, self).__init__()
        self.config = config

        # 输入特征维度映射到模型维度
        # (批量大小, 序列长度, 特征维度)
        # → 嵌入层 → (批量大小, 序列长度, d_model)
        # → 加位置编码 → (批量大小, 序列长度, d_model)
        # → 编码器层（N次） → (批量大小, 序列长度, d_model)
        # → 输出层 → (批量大小, 预测长度, 目标维度)
        self.input_projection = nn.Linear(config['input_dim'], config['d_model'])

        # 位置编码
        self.pos_encoder = PositionalEncoding(config['d_model'], config['max_len'])

        # Transformer编码层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['n_head'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation=config['activation']
        )

        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=config['num_layers']
        )

        # 输出层，预测未来价格
        self.output_layer = nn.Linear(
            config['d_model'],
            config['output_dim'] * config['prediction_length']
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        initrange = 0.1
        self.input_projection.bias.data.zero_()
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        参数:
            src: 输入序列，形状为 (batch_size, seq_len, input_dim)

        返回:
            output: 预测结果，形状为 (batch_size, prediction_length, output_dim)
        """

        # 调整维度以适应Transformer的输入要求 (seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, input_dim)

        # 投影到模型维度
        src = self.input_projection(src)  # (seq_len, batch_size, d_model)

        # 添加位置编码
        src = self.pos_encoder(src)  # (seq_len, batch_size, d_model)

        # Transformer编码
        output = self.transformer_encoder(src)  # (seq_len, batch_size, d_model)

        # 使用最后一个时间步的输出进行预测
        last_output = output[-1, :, :]  # (batch_size, d_model)

        # 输出层
        prediction = self.output_layer(last_output)  # (batch_size, prediction_length * output_dim)

        # 调整形状为 (batch_size, prediction_length, output_dim)
        prediction = prediction.view(
            -1,
            self.config['prediction_length'],
            self.config['output_dim']
        )

        return prediction


class TransformerWithDecoderModel(nn.Module):
    """包含编码器和解码器的Transformer价格预测模型"""

    def __init__(self):
        super().__init__()
        self.config = config

        # 输入特征维度映射到模型维度
        self.input_projection = nn.Linear(config['input_dim'], config['d_model'])
        self.target_projection = nn.Linear(config['output_dim'], config['d_model'])

        # 位置编码
        self.pos_encoder = PositionalEncoding(
            d_model=config['d_model'],
            dropout=config['dropout']
        )

        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation=config['activation']
        )

        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=config['num_layers']
        )

        # Transformer解码器层
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation=config['activation']
        )

        # Transformer解码器
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers,
            num_layers=config['num_layers']
        )

        # 输出层
        self.output_layer = nn.Linear(config['d_model'], config['output_dim'])

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        initrange = 0.1
        self.input_projection.bias.data.zero_()
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.target_projection.bias.data.zero_()
        self.target_projection.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        """
        参数:
            src: 输入序列，形状为 (batch_size, seq_len, input_dim)
            tgt: 目标序列（用于教师强制），形状为 (batch_size, pred_len, output_dim)

        返回:
            output: 预测结果，形状为 (batch_size, prediction_length, output_dim)
        """
        # 调整维度 (seq_len, batch_size, dim)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, input_dim)
        tgt = tgt.permute(1, 0, 2)  # (pred_len, batch_size, output_dim)

        # 投影到模型维度
        src = self.input_projection(src)  # (seq_len, batch_size, d_model)
        tgt = self.target_projection(tgt)  # (pred_len, batch_size, d_model)

        # 添加位置编码
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # 编码器输出
        memory = self.transformer_encoder(src)  # (seq_len, batch_size, d_model)

        # 解码器输出
        output = self.transformer_decoder(tgt, memory)  # (pred_len, batch_size, d_model)

        # 输出层
        output = self.output_layer(output)  # (pred_len, batch_size, output_dim)

        # 调整回原始维度 (batch_size, pred_len, output_dim)
        output = output.permute(1, 0, 2)

        return output

def get_transformer_model():
    """根据配置获取相应的Transformer模型"""
    if config['model_type'] == 'encoder': # 这里仅仅使用包含编码器的encoder
        return TransformerEncoderModel()
    elif config['model_type'] == 'decoder':
        return TransformerWithDecoderModel()
    else:
        raise ValueError(f"不支持的模型类型: {config['model_type']}")
