import torch
from torch import nn
from torch import nn
import torch.nn.functional as F
from common import utils

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        """
        :param d_model: 特征维度
        :param nhead: 注意力头数
        :param dropout: dropout 比例
        """
        super(CustomTransformerEncoderLayer, self).__init__()
        # 使用 nn.MultiheadAttention，batch_first=True 便于使用形状 (B, seq_len, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 前馈网络
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        # 两个 LayerNorm 层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        :param src: 输入张量，形状 (B, seq_len, d_model)
        :return:
            - src: Transformer Encoder 层输出 (B, seq_len, d_model)
            - attn_weights: 注意力权重，形状 (B, num_heads, seq_len, seq_len)
        """
        # 计算自注意力，并返回注意力权重
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True
        )
        # 残差连接 + LayerNorm
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        # 前馈网络部分
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src, attn_weights


class ExtractionModule(nn.Module):
    def __init__(self, mode, batch_size, grid_shape, mask_size, embed_dim, num_heads):
        super().__init__()
        self.mode = mode
        if mode == 'discrete':
            self.input_channel = 11
            self.action_embedding = nn.Embedding(7, embed_dim)
        else:
            self.input_channel = grid_shape[0]
            self.action_fc = nn.Linear(1, embed_dim)

        self.batch_size = batch_size
        self.mask_size = mask_size
        self.y, self.x = mask_size // 2, mask_size // 2
        self.conv1 = nn.Conv2d(self.input_channel, embed_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

        # 2. 展平操作，将 (B, embed_dim, H, W) 展平为 (B, embed_dim, H*W)
        self.flatten = nn.Flatten(2)
        # 位置编码：为每个 patch 学习一个位置编码，形状为 (1, height*width, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, mask_size * mask_size, embed_dim))

        # 3. 动作嵌入：将离散动作编码为与 embed_dim 相同的向量
        self.fuse_fc = nn.Linear(embed_dim * 2, embed_dim)

        # 4. 多层自定义 Transformer Encoder 层
        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            for _ in range(1)
        ])

    def forward(self, state, action):
        B, _, H, W = state.size()
        assert B == self.batch_size, "batch_size mismatch!"
        
        if self.mode == 'discrete':
            obj = state[:, 0, :, :]
            color = state[:, 1, :, :]
            dir = state[:, 2, :, :]
            obj = F.one_hot(obj.reshape(B, -1).long(), num_classes=4)
            color = F.one_hot(state[:, 1, :, :].reshape(B, -1).long(), num_classes=3)
            dir = F.one_hot(state[:, 2, :, :].reshape(B, -1).long(), num_classes=4)
            state_emb = torch.cat([obj, color, dir], dim=-1).float()
            state_emb = state_emb.transpose(1,2).reshape(B, self.input_channel, H, W)
            action_emb = self.action_embedding(action)

        else:
            action_emb = self.action_fc(action.unsqueeze(1))  # (B, embed_dim)
            state_emb = state



        x = self.relu(self.bn1(self.conv1(state_emb)))
        x = self.relu(self.bn2(self.conv2(x)))
        # 2. 展平空间维度，将 (B, embed_dim, 5, 5) 转换为 (B, embed_dim, 25)
        x = self.flatten(x)
        # 转置为 (B, 25, embed_dim) 作为 Transformer 的输入
        x = x.transpose(1, 2)
        # 3. 添加位置编码
        x = x + self.pos_embedding  # (B, 25, embed_dim)

        # 4. 融合动作信息
        # 假设 action 为离散变量，shape (B,)
        # 将 action_emb 扩展至 (B, 25, embed_dim)（对每个 token都添加相同的动作信息）
        action_emb = action_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        fused = torch.cat([x, action_emb], dim=-1)  # (B, 25, embed_dim*2)
        x = self.fuse_fc(fused)  # (B, 25, embed_dim)

        # 5. 依次通过 Transformer Encoder 层
        attn_weights = None
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)

        return x, attn_weights
    


class PredictionModule(nn.Module):
    def __init__(self, embed_dim, delta_shape, hidden_dim=128):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 3)


    def forward(self, x):
        x = self.fc(x)
        x = x.transpose(1, 2).flatten(1)
        return x
    


