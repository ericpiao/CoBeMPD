import torch
import torch.nn as nn
from transformers import AutoModel

class MLPModel(nn.Module):
    def __init__(self, bert_model, hidden_dim, classifier_dims):
        super(MLPModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model)

        # 实例级分类器：每个实例得到一个得分 s_i ∈ [0,1]
        self.instance_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出为单个数值
            nn.Sigmoid()
        )
        # 跨实例注意力模块
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.bert.config.hidden_size,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.classifier_layers = torch.nn.Linear(20,1)


    def forward(self, block_ids, line_attention_mask,file_mask):
        batch_size, num_instances, seq_len = block_ids.shape

        # Flatten for BERT
        flat_ids = block_ids.view(-1, seq_len)
        flat_mask = line_attention_mask.view(-1, seq_len)

        outputs = self.bert(input_ids=flat_ids, attention_mask=flat_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B*N, H]
        instance_embeddings = cls_embeddings.view(batch_size, num_instances, -1)  # [B, N, H]
        instance_embeddings = self.transformer_encoder(instance_embeddings)
        # Instance-level scores
        instance_logits = self.instance_classifier(instance_embeddings)  # [B, N, 1]
        instance_scores = instance_logits.squeeze(-1)                    # [B, N], s_i
        masked_scores = instance_scores * file_mask
        # Bag-level prediction by max pooling
        bag_pred = self.classifier_layers(masked_scores).squeeze()
        return bag_pred,masked_scores  # 分别是 bag-level 和 instance-level 预测
