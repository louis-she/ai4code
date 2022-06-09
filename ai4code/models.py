import torch
from transformers import AlbertForSequenceClassification, AutoModel, BertForSequenceClassification, DistilBertConfig, DistilBertForSequenceClassification, DistilBertModel, LongformerModel, PretrainedConfig, RobertaForSequenceClassification
import torch.nn as nn
import transformers


class Model(nn.Module):

    def __init__(self, pretrained_path, dropout=0.2):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.backbone = AutoModel.from_pretrained(pretrained_path)
        try:
            out_features_num = self.backbone.encoder.layer[-1].output.dense.out_features
        except:
            out_features_num = 768
        output_features_num += 1

        self.classifier = nn.Sequential(
            nn.Linear(in_features=out_features_num, out_features=out_features_num),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=out_features_num, out_features=1),
        )

    def forward(self, x, mask, code_ratio):
        output = self.backbone(x, mask)
        x = output[0]  # (bs, seq_len, dim)
        x = x[:, 0] # (bs, dim)
        x = torch.cat([x, code_ratio], dim=1) # (bs, dim + 1)
        x = self.classifier(x)
        return x


class MultiHeadModel(nn.Module):

    def __init__(self, pretrained_path, dropout=0.2):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.backbone = AutoModel.from_pretrained(pretrained_path)
        try:
            out_features_num = self.backbone.encoder.layer[-1].output.dense.out_features
        except:
            out_features_num = 768

        self.classifier = nn.Sequential(
            nn.Linear(in_features=out_features_num, out_features=out_features_num),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=out_features_num, out_features=1),
        )

        self.ranker = nn.Sequential(
            nn.Linear(in_features=out_features_num, out_features=out_features_num),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=out_features_num, out_features=1),
        )

    def forward(self, x, mask, cell_nums):
        output = self.backbone(x, mask)
        x = output[0]  # (bs, seq_len, dim)
        feature = x[:, 0] # (bs, dim)
        # x = torch.cat([x, cell_nums], dim=1) # (bs, dim + 2)
        return self.classifier(feature), self.ranker(feature)


class LongFormer(nn.Module):
    def __init__(self, pretrained_path, dropout=0.2):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.backbone = AutoModel.from_pretrained(pretrained_path)
        try:
            out_features_num = self.backbone.encoder.layer[-1].output.dense.out_features
        except:
            out_features_num = 768

        self.classifier = nn.Sequential(
            nn.Linear(in_features=out_features_num, out_features=out_features_num),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=out_features_num, out_features=1),
        )

    def forward(self, x, mask, cell_mask):
        output = self.backbone(x, mask, global_attention_mask=cell_mask)
        x = output[0]  # (bs, seq_len, dim)
        x = x[:, 0]
        x = self.classifier(x)
        return x
