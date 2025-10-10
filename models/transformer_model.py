# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import config


class PositionalEncoding(nn.Module):
    """位置编码模块，为输入序列添加位置信息"""

    def __init__(self, d_model, max_len, dropout=0.1):
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
        :param x:输入张量，根据batch_first不同有两种形状:
                 - batch_first=True时: (batch_size, seq_len, d_model)
                 - batch_first=False时: (seq_len, batch_size, d_model)
        :return:
        """
        # 根据输入形状自动适配位置编码添加方式
        if x.dim() == 3 and x.size(1) < x.size(0):  # 判断是否为(batch, seq, dim)格式
            x = x + self.pe[:x.size(1)].permute(1, 0, 2)  # 适配batch_first=True
        else:
            x = x + self.pe[:x.size(0)]  # 适配默认格式
        return self.dropout(x)

class TransformerEncoderModel(nn.Module):
    """仅使用Transformer编码器的价格预测模型"""

    def __init__(self):
        super(TransformerEncoderModel, self).__init__()
        self.config = config

        # 输入特征维度映射到模型维度
        self.input_projection = nn.Linear(config['input_dim'], config['d_model'])

        # 位置编码
        self.pos_encoder = PositionalEncoding(config['d_model'], config['max_len'])

        # Transformer编码层 - 关键修改：添加batch_first=True
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['n_head'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation=config['activation'],
            batch_first=True  # 这里添加batch_first=True
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
        # 对所有子模块递归初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 线性层用Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                # 嵌入层用小范围均匀分布
                module.weight.data.uniform_(-0.02, 0.02)

    def forward(self, src):
        """
        参数:
            src: 输入序列，形状为 (batch_size, seq_len, input_dim)

        返回:
            output: 预测结果，形状为 (batch_size, prediction_length, output_dim)
        """
        # 当batch_first=True时，无需转置维度，保持(batch_size, seq_len, input_dim)
        src = self.input_projection(src)  # (batch_size, seq_len, d_model)

        # 添加位置编码（已适配batch_first）
        src = self.pos_encoder(src)  # (batch_size, seq_len, d_model)

        # Transformer编码
        output = self.transformer_encoder(src)  # (batch_size, seq_len, d_model)

        # 关键修改：全局平均池化，融合整个序列的特征（替代仅取最后一步）
        pooled_output = output.mean(dim=1)  # (batch_size, d_model) → 每个样本的序列全局特征
        prediction = self.output_layer(pooled_output)  # (batch_size, 1*output_dim)

        # 调整形状
        prediction = prediction.view(
            -1,
            self.config['prediction_length'],
            self.config['output_dim']
        )

        return prediction


class TransformerWithDecoderModel(nn.Module):
    """包含编码器和解码器的Transformer价格预测模型"""
    # 此类已正确设置batch_first=True，无需修改
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
            activation=config['activation'],
            batch_first=True
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
            activation=config['activation'],
            batch_first=True
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
        # 投影到模型维度（保持batch_first=True的维度顺序）
        src = self.input_projection(src)  # (batch_size, seq_len, d_model)
        tgt = self.target_projection(tgt)  # (batch_size, pred_len, d_model)

        # 添加位置编码
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # 编码器输出
        memory = self.transformer_encoder(src)  # (batch_size, seq_len, d_model)

        # 解码器输出
        output = self.transformer_decoder(tgt, memory)  # (batch_size, pred_len, d_model)

        # 输出层
        output = self.output_layer(output)  # (batch_size, pred_len, output_dim)

        return output

def get_transformer_model():
    """根据配置获取相应的Transformer模型"""
    if config['model_type'] == 'encoder':
        return TransformerEncoderModel()
    elif config['model_type'] == 'decoder':
        return TransformerWithDecoderModel()
    else:
        raise ValueError(f"不支持的模型类型: {config['model_type']}")
