import torch
from torch import nn
from torch import nn
import torch.nn.functional as F
from enum import Enum

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=128, dropout=0.2):
        super().__init__()
        # 1) Multi-Head Attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        # 残差 & LayerNorm
        self.ln1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        # 2) 前馈网络 (FFN): 通常是 Linear -> ReLU -> Linear
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_dim),
        )
        # 残差 & LayerNorm
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, Q, KV, attn_mask=None):
        attn_out, attn_weights = self.attn(query=Q, key=KV, value=KV, attn_mask=attn_mask)
        # 残差连接
        Q = Q + self.dropout1(attn_out)
        # LayerNorm
        Q = self.ln1(Q)
        # ---- (2) 前馈网络 (FFN) ----
        ffn_out = self.ffn(Q)
        # 残差连接
        Q = Q + self.dropout2(ffn_out)
        # LayerNorm
        out = self.ln2(Q)
        return out, attn_weights

def generate_positional_encoding(position, embed_dim):
    position_encoding = torch.zeros(position.size(0), embed_dim, device=position.device)
    for i in range(embed_dim // 2):
        position_encoding[:, 2 * i] = torch.sin(position[:, 0] / (10000 ** (2 * i / embed_dim)))
        position_encoding[:, 2 * i + 1] = torch.cos(position[:, 0] / (10000 ** (2 * i / embed_dim)))
    return position_encoding

class ExtractionModule(nn.Module):
    def __init__(self, mode, batch_size, grid_shape, embed_dim, num_heads, ff_dim=128, dropout=0):
        super().__init__()
        ff_dim = 2 * embed_dim
        self.mode = mode
        self.channel, self.row, self.col = grid_shape
        self.batch_size = batch_size
        dim = 4
        if mode == 'conv' or mode == 'mlp':
            if mode == 'conv':
                self.conv = nn.Sequential(
                    nn.Conv2d(self.channel, embed_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            if mode == 'mlp':
                self.mlp = nn.Sequential(
                    nn.Linear(self.channel, embed_dim),
                    nn.ReLU()
                )
            
            self.relative_pos_fc = nn.Linear(2, dim)
            self.dir_embedding = nn.Embedding(4, dim)    # 方向4种 -> 4维
            self.act_embedding = nn.Embedding(7, dim)    # 动作7种 -> 4维
            self.pos_fc = nn.Linear(2, dim)  
            self.query_fc = nn.Linear(2 * dim, embed_dim)
            self.kv_fc = nn.Linear(embed_dim + dim, embed_dim)

            self.transformer_block = TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout
            )
            # 生成网格并归一化
            row_idx = torch.arange(self.row, dtype=torch.float32)
            col_idx = torch.arange(self.col, dtype=torch.float32)
            row_grid, col_grid = torch.meshgrid(row_idx, col_idx, indexing='ij')

            row_grid = row_grid / (self.row - 1)   # 归一化 [0, 1]
            col_grid = col_grid / (self.col - 1)
            position_norm = torch.stack([row_grid, col_grid], dim=-1)  # [row, col, 2]
            position_norm = position_norm.reshape(-1, 2)               # [row*col, 2]

            self.register_buffer(
                'batch_position', 
                position_norm.unsqueeze(0).expand(batch_size, -1, -1)
            )
        
        if mode == 'fc':
            self.fc_state = nn.Sequential(
                        nn.Linear(grid_shape[0] * grid_shape[1] * grid_shape[2], 4 * embed_dim),
                        nn.LayerNorm(4 * embed_dim), 
                        nn.LeakyReLU(negative_slope=0.01), 
                        
                        nn.Linear(4 * embed_dim, 2 * embed_dim),
                        nn.LayerNorm(2 * embed_dim),  
                        nn.LeakyReLU(negative_slope=0.01),  
                        
                        nn.Linear(2 * embed_dim, embed_dim),
                        nn.LayerNorm(embed_dim),  
                        nn.LeakyReLU(negative_slope=0.01), 
                        
                        nn.Linear(embed_dim, embed_dim),
                        nn.LayerNorm(embed_dim),  
                        nn.LeakyReLU(negative_slope=0.01)  
                    )
            
            self.act_embedding = nn.Embedding(7, dim)    # 动作7种 -> 4维
            self.fc_fusion = nn.Sequential(
                        nn.Linear(embed_dim + dim, embed_dim),
                        nn.LayerNorm(embed_dim)
                    )
            

        if mode == 'discrete':
            self.transformer_block = TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout
            )
            self.act_layer = nn.Linear(7, embed_dim)
            self.state_layer = nn.Sequential(
                        nn.Linear(11, embed_dim // 2),
                        # nn.LayerNorm(4 * embed_dim), 
                        # nn.LeakyReLU(negative_slope=0.01), 
                        
                        nn.Linear(embed_dim // 2, embed_dim),
                        # nn.LayerNorm(2 * embed_dim),  
                        # nn.LeakyReLU(negative_slope=0.01),  
                        
                        nn.Linear(embed_dim, embed_dim * 2),
                        # nn.LayerNorm(embed_dim),  
                        # nn.LeakyReLU(negative_slope=0.01), 
                        
                        nn.Linear(embed_dim * 2, embed_dim)
                        # nn.LayerNorm(embed_dim),  
                        # nn.LeakyReLU(negative_slope=0.01)  
                        )



    def forward(self, state, action):
        B = state.size(0)
        assert B == self.batch_size, "batch_size mismatch!"

        if self.mode == 'conv' or self.mode == 'mlp':
            if self.mode == 'conv':
                state_embed = self.conv(state)
                state_embed = state_embed.flatten(2).permute(0, 2, 1) 
            
            if self.mode == 'mlp':
                state_embed = state.flatten(2).permute(0, 2, 1) 
                state_embed = self.mlp(state_embed)

            agent_coords = torch.argwhere(state[:, 0, :, :] == 1) 
            agent_coords = agent_coords[agent_coords[:,0].sort()[1]]
            agent_position = agent_coords[:, 1:]  

            agent_pos_norm = torch.zeros_like(agent_position, dtype=torch.float32)
            agent_pos_norm[:, 0] = agent_position[:, 0] / (self.row - 1)
            agent_pos_norm[:, 1] = agent_position[:, 1] / (self.col - 1)
            agent_pos_emb = self.pos_fc(agent_pos_norm)  

            relative_position = self.batch_position - agent_pos_norm.unsqueeze(1)  
            relative_position_emb = self.relative_pos_fc(relative_position)       

            batch_indices = torch.arange(B, device=state.device)
            dir_map = state[:, 2, :, :]  
            y, x = agent_position[:, 0], agent_position[:, 1]
            dir_idx = (dir_map[batch_indices, y, x] * 3).long()      
            dir_emb = self.dir_embedding(dir_idx)                    

            act_idx = (action * 6).long()  
            act_emb = self.act_embedding(act_idx)  

            Q_in = torch.cat([act_emb, dir_emb], dim=-1)  
            Q = self.query_fc(Q_in).unsqueeze(1)                         

            KV_in = torch.cat([state_embed, relative_position_emb], dim=-1) 
            KV = self.kv_fc(KV_in)  

            attention_output, attention_weights = self.transformer_block(Q, KV)
            output = attention_output.squeeze(1)
            atten_weights = attention_weights.squeeze(1)

        
        if self.mode == 'fc':
            state = state.reshape(B, -1)
            state_emb = self.fc_state(state)
            act_idx = (action * 6).long()  
            act_emb = self.act_embedding(act_idx)  
            output = self.fc_fusion(torch.cat([state_emb, act_emb], dim=-1) )
            atten_weights = torch.zeros((B, self.row * self.col))

        if self.mode == 'discrete':
            obj = state[:, 0, :, :]
            color = state[:, 1, :, :]
            dir = state[:, 2, :, :]

            obj = F.one_hot(obj.reshape(B, -1).long(), num_classes=4)
            color = F.one_hot(state[:, 1, :, :].reshape(B, -1).long(), num_classes=3)
            dir = F.one_hot(state[:, 2, :, :].reshape(B, -1).long(), num_classes=4)
            state_emb = torch.cat([obj, color, dir], dim=-1).float()
            act_emb = F.one_hot(action, num_classes=7).float()

            state_emb = self.state_layer(state_emb)
            act_emb = self.act_layer(act_emb)

            attention_output, attention_weights = self.transformer_block(act_emb.unsqueeze(1), state_emb)
            output = attention_output.squeeze(1)
            atten_weights = attention_weights.squeeze(1)
            
            
        return output, atten_weights

class PredictionModule(nn.Module):
    def __init__(self, embed_dim, grid_shape, hidden_dim=128):
        super().__init__()
        hidden_dim = 4 * embed_dim  # 增大隐藏层的维度

        # 增加更多的全连接层
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, grid_shape[0] * grid_shape[1] * grid_shape[2])

        # 批归一化
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.bn3 = nn.LayerNorm(hidden_dim // 2)

        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.3)

    def forward(self, extracted_features):
        # 第一层
        x = F.leaky_relu(self.bn1(self.fc1(extracted_features)))
        x = self.dropout(x)
        
        # 第二层
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        # 第三层
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        # 输出层
        output = self.fc4(x)
        return F.softplus(output)  # 使用 softplus 激活输出
    


