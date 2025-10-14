# -*- coding: utf-8 -*-

import math
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from config import config

logger = logging.getLogger("transformer_model")


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
                 形状为(batch_size, seq_len, d_model)
                 - batch_first=True时: (batch_size, seq_len, d_model)
                 - batch_first=False时: (seq_len, batch_size, d_model)
        :return:

        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].transpose(0, 1)  # 适配batch_first=True
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """手动实现硕放点积注意力"""

    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        """
        args:
            q:形状(batch_size,n_heads,seq_len_q,d_model_k)
            k:形状(batch_size,n_heads,seq_len_k,d_model_k)
            v:形状(batch_size,n_heads,seq_len_v,d_model_v)
            mask:形状(batch_size,1,seq_len_q,seq_len_v) 或 None

        Returns:
            output: 注意力输出，形状 (batch_size, n_heads, seq_len_q, d_v)
            attn: 注意力权重，形状 (batch_size, n_heads, seq_len_q, seq_len_k)
        """

        d_model_k = q.size(-1)

        # q:(batch_size,n_heads,seq_len_q,d_model_k)
        # k.transpose(-2,-1):(batch_size,n_heads,d_model_k,seq_len_k)
        # scores:(batch_size,n_heads,seq_len_q,seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model_k)

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        # attn:(batch_size,n_heads,seq_len_q,seq_len_k)
        # v   :(batch_size,n_heads,seq_len_v,d_model_v)
        # seq_len_k = seq_len_v
        output = torch.matmul(attn, v)
        # output :(batch_size,n_heads,seq_len_q,d_model_v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """手动实现多头注意力"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # 确保模型维度可以被heads头数整除
        assert self.d_k * n_heads == d_model, "d_model must be divisible by n_heads"

        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 输出线性层
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        args:
            q:形状(batch_size,seq_len_q,d_model)
            k:形状(batch_size,seq_len_k,d_model)
            v:形状(batch_size,seq_len_v,d_model)
            mask:形状(batch_size,seq_len_q,seq_len_k) 或 None

        Returns:
            output:形状(batch_size,seq_len_q,d_model)
            attn:形状(batch_size,n_heads,seq_len_q,seq_len_k)
        """
        batch_size = q.size(0)

        # 线性变换并分成多头
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                               2)  # (batch_size,n_heads,seq_len_q,d_k)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                               2)  # (batch_size, n_heads, seq_len_k, d_k)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                               2)  # (batch_size, n_heads, seq_len_v, d_v)

        # 调整掩码形状以适应多头
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size,1,seq_len_q,seq_len_k)

        # 计算注意
        # output:(batch_size, n_heads, seq_len_q, d_v)
        # attn: (batch_size, n_heads, seq_len_q, seq_len_k)
        output, attn = self.attention(q, k, v, mask=mask)

        # 拼接多头结果
        # 其中d_model=seq_len_q*d_v
        output = output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                          self.d_model)  # (batch_size, seq_len_q, d_model)

        # 输出线性层
        output = self.w_o(output)
        output = self.dropout(output)

        return output, attn


class PositionWiseFeedForward(nn.Module):
    """wise前馈网络"""

    def __init__(self, d_model, dim_feedforward, dropout=0.1, activation='relu'):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

        # 激活函数
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        """
        args:
            x:形状(batch_size,seq_len,d_model)

        Returns:
            形状(batch_size,seq_len,d_model)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """手动实现Transformer编码器层"""

    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, dim_feedforward, dropout, activation)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        args:
            x:形状(batch_size,seq_len,d_model)
            src_mask:注意力掩码

        Returns:
            形状(batch_size,seq_len,d_model)
        """
        # 自注意力子层
        # attn_output : (batch_size,seq_len_q,d_model)
        attn_output, _ = self.self_attn(x, x, x, mask=src_mask)  # 在编码其中，q k v是来自同一个输入序列
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x  # x (batch_size,seq_len_q,d_model)


class TransformerDecoderLayer(nn.Module):
    """手动实现Transformer解码器层"""

    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1, activation='relu'):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)  # 掩码自注意力
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)  # 编码器-解码器注意力
        self.feed_forward = PositionWiseFeedForward(d_model, dim_feedforward, dropout, activation)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        args:
            tgt:目标序列，形状(batch_size,tgt_seq_len,d_model)
            memory:编码器输出，形状(batch_size,src_seq_len,d_model)
            tgt_mask:目标序列掩码
            memory_mask:编码器-解码器注意力掩码

        Returns:
            形状(batch_size,tgt_seq_len,d_model)
        """
        # 掩码自注意力层
        attn_output, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        # 编码器——解码器注意力子层
        attn_output, _ = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(attn_output)
        tgt = self.norm2(tgt)

        # 前馈网络子层
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(ff_output)
        tgt = self.norm3(tgt)

        return tgt


class TransformerEncoder(nn.Module):
    """实现Transformer编码器"""

    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, x, mask=None):
        """
        args:
            x:输入序列，形状(batch_size,seq_len,d_model)
            mask:注意力掩码

        Returns:
            形状(batch_size,seq_len,d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoder(nn.Module):
    """实现Transformer解码器"""

    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        args:
             tgt:目标序列，形状(batch_size,tgt_seq_len,d_model)
             mmemory:输出器输出,形状(batch_size,src_seq_len,d_model)
             tagt_mask:目标序列掩码
             memory_mask:编码器-解码器注意力掩码

         Returns:
             形状(batch_size,tgt_seq_len,d_model)
        """
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return tgt


class TransformerEncoderModel(nn.Module):
    """仅使用Transformer编码器的价格预测模型"""

    def __init__(self):
        super(TransformerEncoderModel, self).__init__()

        # 输入特征维度映射到模型维度
        self.input_projection = nn.Linear(config['input_dim'], config['d_model'])

        # 位置编码
        self.pos_encoder = PositionalEncoding(config['d_model'], config['max_len'])

        # 创建手动实现的Transformer编码层
        encoder_layer = TransformerEncoderLayer(
            d_model=config['d_model'],
            n_heads=config['n_head'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation=config['activation']
        )

        # Transformer编码器
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
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
        logger.info(f"对所有子模块递归初始化")
        for module in self.modules():
            if isinstance(module, nn.Linear):
                logger.info(f"线性层用Xavier初始化")
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                logger.info(f"嵌入层用小范围均匀分布")
                module.weight.data.uniform_(-0.02, 0.02)

    def forward(self, src):
        """
        args:
            src:输入序列形状(batch_size,src_seq_len,d_model)

        returns:
            output:预测结果,形状为(batch_size,prediction_length,output_dim)
        """
        src = self.input_projection(src)  # (batch_size,seq_len,d_model)
        src = self.pos_encoder(src)  # 添加位置编码
        output = self.transformer_encoder(src)  # 编码器输出

        # 取最后一个时间步的输出来进行预测
        last_step_output = output[:, -1, :]
        prediction = self.output_layer(last_step_output)

        # 调整形状
        prediction = prediction.view(
            -1,
            config['prediction_length'],
            config['output_dim']
        )

        return prediction


class TransformerWithDecoderModel(nn.Module):
    """编码器和解码器的Transformer价格预测模型"""

    def __init__(self):
        super(TransformerWithDecoder, self).__init__()

        # 输入特征维度映射到模型维度
        self.input_projection = nn.Linear(config['input_dim'], config['d_model'])
        self.target_projection = nn.Linear(config['output_dim'], config['d_model'])

        # 位置编码
        self.pos_encoder = PositionalEncoding(
            d_model=config['d_model'],
            max_len=config['max_len'],
            dropout=config['dropout']
        )

        # 创建Transformer编码器层
        encoder_layer = TransformerEncoderLayer(
            d_model=config['d_model'],
            n_heads=config['n_head'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation=config['activation']
        )

        # Transformer编码器
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=config['num_layers']
        )

        # 创建Transformer解码器层
        decoder_layer = TransformerDecoderLayer(
            d_model=config['d_model'],
            n_heads=config['n_head'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation=config['activation']
        )

        # Transformer解码器
        self.transformer_decoder=TransformerDecoder(
            decoder_layer,
            num_layers=config['num_layers']
        )

        # 输出层
        self.output_layer=nn.Linear(config['d_model'], config['output_dim'])

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
        # 投影到模型维度
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
