import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class MILMaxPool(nn.Module):
    """
    多实例学习 (MIL) + MaxPooling 聚合（无置信度加权）
    输入:
        block_ids: [B, N, L]
        line_attention_mask: [B, N, L]
        file_mask: [B, N]
    输出:
        bag_pred: 文件级预测
        instance_scores: 实例级恶意分数
    """
    def __init__(self, bert_model, hidden_dim=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)

        # Transformer 聚合实例间信息
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.bert.config.hidden_size,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # 实例分类器（无 dropout）
        self.instance_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, block_ids, line_attention_mask, file_mask):
        B, N, L = block_ids.shape

        # === [1] BERT 编码 ===
        flat_ids = block_ids.view(-1, L)
        flat_mask = line_attention_mask.view(-1, L)
        outputs = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]   # [B*N, H]

        instance_embeddings = cls_embeddings.view(B, N, -1)   # [B, N, H]

        # === [2] 跨实例关系建模 ===
        instance_embeddings = self.transformer_encoder(instance_embeddings)

        # === [3] 实例级恶意得分 ===
        logits = self.instance_classifier(instance_embeddings).squeeze(-1)  # [B, N]
        instance_scores = logits * file_mask

        # === [4] 文件级预测 (MaxPooling 聚合) ===
        masked_scores = instance_scores.masked_fill(file_mask == 0, -1e9)
        bag_pred, _ = torch.max(masked_scores, dim=1)        # [B]

        # 可保持你之前的小变换（让分布更平滑）
        bag_pred = torch.sigmoid(2 * bag_pred - 1)

        return bag_pred, instance_scores
