# adaptive_vit.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

class LatencyEncoder(nn.Module):
    """将标量延迟预算编码为延迟令牌"""
    def __init__(self, embed_dim, hidden_dim=256, max_budget_levels=100):
        super().__init__()
        self.embed_dim = embed_dim
        # 预计算一个足够大的正弦编码表，比如100个级别
        self.register_buffer('positional_encoding', self._get_sinusoid_encoding_table(max_budget_levels, hidden_dim))
        
        # 使用MLP将编码后的向量转换为最终的延迟令牌
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """创建正弦位置编码表"""
        def get_position_angle_vec(position):
            return [position / math.pow(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = torch.tensor([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return sinusoid_table.float()

    def forward(self, budget):
        # budget: (batch_size, 1)
        # 将 budget (0-1) 缩放到编码表的索引范围
        budget_scaled = budget * (self.positional_encoding.shape[0] - 1)
        
        # 使用最近邻的索引来获取编码
        indices = budget_scaled.round().long().squeeze(-1)
        
        # 从编码表中获取编码
        encoded_budget = self.positional_encoding[indices]
        latency_token = self.mlp(encoded_budget)
        return latency_token.unsqueeze(1) # -> (batch_size, 1, embed_dim)

class AdaptiveViT(nn.Module):
    def __init__(self, timm_model_name='vit_base_patch16_224', num_classes=100, pretrained=True, num_adaptive_layers=6):
        super().__init__()
        self.base_model = timm.create_model(timm_model_name, pretrained=pretrained, num_classes=num_classes)
        
        embed_dim = self.base_model.embed_dim
        self.num_total_layers = len(self.base_model.blocks)
        self.num_adaptive_layers = num_adaptive_layers
        self.num_fixed_layers = self.num_total_layers - self.num_adaptive_layers

        # 初始化延迟编码器和调度器
        self.latency_encoder = LatencyEncoder(embed_dim)
        # 调度器是一个简单的线性层，输出每个自适应层的 logits
        self.scheduler = nn.Linear(embed_dim, self.num_adaptive_layers)
        
        # 将 ViT 的 blocks 分成固定和自适应两部分
        self.fixed_blocks = self.base_model.blocks[:self.num_fixed_layers]
        self.adaptive_blocks = self.base_model.blocks[self.num_fixed_layers:]

        # --- 关键修正：扩展位置编码以适应 latency_token ---
        # 存储原始的位置编码
        original_pos_embed = self.base_model.pos_embed # shape (1, 197, 768)
        
        # 创建一个新的、更大的位置编码参数
        new_pos_embed_shape = (1, original_pos_embed.shape[1] + 1, original_pos_embed.shape[2])
        self.pos_embed = nn.Parameter(torch.zeros(new_pos_embed_shape))
        
        # 使用预训练模型的值来初始化新的位置编码
        with torch.no_grad():
            # 新的位置编码序列：[latency_pos, cls_pos, patch_pos]
            # 第0个位置给 latency_token，可以初始化为0
            self.pos_embed[:, 0] = 0
            # 第1个位置给 cls_token，从原始编码的第0个位置拷贝
            self.pos_embed[:, 1] = original_pos_embed[:, 0]
            # 后续位置给 patch_tokens，从原始编码的第1个位置及之后拷贝
            self.pos_embed[:, 2:] = original_pos_embed[:, 1:]
        
        # 将 base_model 中的 pos_embed 设为 None，避免在后续使用中产生混淆
        self.base_model.pos_embed = None


    def forward(self, x, budget):
        # budget: (batch_size,)
        batch_size = x.shape[0]
        budget = budget.unsqueeze(-1) # -> (batch_size, 1)

        # 1. 编码预算为 latency_token
        latency_token = self.latency_encoder(budget)
        
        # 2. 准备ViT输入序列
        x = self.base_model.patch_embed(x)
        cls_token = self.base_model.cls_token.expand(batch_size, -1, -1)
        # 拼接: [latency_token, cls_token, patch_tokens]
        x = torch.cat((latency_token, cls_token, x), dim=1)
        
        # --- 关键修正：使用我们自己创建的、尺寸正确的 pos_embed ---
        x = x + self.pos_embed

        # 3. 通过固定层
        for block in self.fixed_blocks:
            x = block(x)

        # 4. 调度器决策
        latency_token_repr = x[:, 0]
        switch_logits = self.scheduler(latency_token_repr)

        # 5. 通过自适应层
        # 计算每个样本需要保留的层数 k
        k_float = (budget.squeeze(-1) * self.num_adaptive_layers)
        k = k_float.ceil().long()
        # 确保 k 至少为1，避免在预算极低时没有层被选中
        k = torch.max(k, torch.ones_like(k))

        if self.training:
            # 训练时：根据预算 k 和 logits 确定要激活的层
            # Gumbel-Softmax 提供可导的随机性
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(switch_logits)))
            perturbed_logits = switch_logits + gumbel_noise
            
            active_indices = torch.zeros_like(switch_logits)
            for i in range(batch_size):
                num_to_keep = k[i]
                topk_indices = torch.topk(perturbed_logits[i], num_to_keep).indices
                active_indices[i].scatter_(0, topk_indices, 1)

        else:
            # 推理时：选择 top-k 最可能的层 (确定性)
            active_indices = torch.zeros_like(switch_logits)
            for i in range(batch_size):
                num_to_keep = k[i]
                if num_to_keep > 0:
                    topk_indices = torch.topk(switch_logits[i], num_to_keep).indices
                    active_indices[i].scatter_(0, topk_indices, 1)
        
        # 提取需要计算的 tokens (cls_token 和 patch_tokens)
        tokens_to_process = x[:, 1:]
        for i, block in enumerate(self.adaptive_blocks):
            # 获取当前batch中哪些样本需要计算此层
            active_mask = active_indices[:, i].bool()
            if active_mask.any():
                sub_batch = tokens_to_process[active_mask]
                processed_sub_batch = block(sub_batch)
                # 将计算结果放回原位
                tokens_to_process[active_mask] = processed_sub_batch
        
        # 6. 分类
        x = self.base_model.norm(tokens_to_process)
        cls_output = x[:, 0]
        output = self.base_model.head(cls_output)
        
        return output