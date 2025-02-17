import torch
from torch import nn
from torch import nn
import torch.nn.functional as F


class EmbeddingModule(nn.Module):
    def __init__(self, data_type, grid_shape, mask_size, embed_dim, num_heads):
        super().__init__()
        self.data_type = data_type
        if data_type == 'discrete':
            self.input_channel = 21
            self.action_embedding = nn.Embedding(7, embed_dim)
        else:
            self.input_channel = grid_shape[0]
            self.action_fc = nn.Linear(1, embed_dim)

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

        self.pre_fc1 = nn.Linear(embed_dim, 2 * embed_dim)
        self.pre_fc2 = nn.Linear(2 * embed_dim, embed_dim)
        self.pre_fc3 = nn.Linear(embed_dim, 3)

    def forward(self, state, action):
        orginal_dim = state.ndim
        if orginal_dim == 3:  # 单个样本
            state = state.unsqueeze(0)  # 变为 (1, C, H, W)
            action = torch.tensor([action]).to(state.device)
        B, C, H, W = state.size()
        
        if self.data_type == 'discrete':
            obj = state[:, 0, :, :]
            color = state[:, 1, :, :]
            dir = state[:, 2, :, :]
            obj = F.one_hot(obj.reshape(B, -1).long(), num_classes=11)
            color = F.one_hot(state[:, 1, :, :].reshape(B, -1).long(), num_classes=6)
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

        # 5. 预测
        x = self.pre_fc1(x)
        x = self.pre_fc2(x)
        x = self.pre_fc3(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        if orginal_dim == 3:
            x = x.squeeze(0)
        return x, None
    

    



    


