import torch
from transformers import AlbertForSequenceClassification, AutoModel, BertForSequenceClassification, DistilBertConfig, DistilBertForSequenceClassification, DistilBertModel, LongformerModel, PretrainedConfig, RobertaForSequenceClassification
import torch.nn as nn
import transformers


class Model(nn.Module):

    def __init__(self, pretrained_path, dropout, with_code_percent_feature):
        super().__init__()
        self.with_code_percent_feature = with_code_percent_feature
        self.pretrained_path = pretrained_path
        self.backbone = AutoModel.from_pretrained(pretrained_path)

        try:
            out_features_num = self.backbone.encoder.layer[-1].output.dense.out_features
        except:
            out_features_num = 768

        if with_code_percent_feature:
            out_features_num += 1

        self.classifier = nn.Sequential(
            nn.Linear(in_features=out_features_num, out_features=out_features_num),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=out_features_num, out_features=1),
        )

    def forward(self, x, mask, cell_numbers):
        output = self.backbone(x, mask)
        x = output[0]  # (bs, seq_len, dim)
        x = x[:, 0] # (bs, dim)
        if self.with_code_percent_feature:
            code_percent = cell_numbers[:, 0] / (cell_numbers[:, 0] + cell_numbers[:, 1])
            x = torch.cat([x, code_percent.unsqueeze(1)], dim=1) # (bs, dim + 1)
        x = self.classifier(x)
        return x


class MultiHeadModel(nn.Module):

    def __init__(self, pretrained_path, with_lm=False, dropout=0.2, with_lstm=False):
        super().__init__()
        self.pretrained_path = pretrained_path
        try:
            self.backbone = AutoModel.from_pretrained(pretrained_path, add_pooling_layer=False)
        except TypeError:
            self.backbone = AutoModel.from_pretrained(pretrained_path)
        self.config = self.backbone.config
        self.with_lm = with_lm
        self.with_lstm = with_lstm

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

        if with_lm:
            self.lm = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.GELU(),
                nn.LayerNorm(self.config.hidden_size),
                nn.Linear(self.config.hidden_size, 1),
            )

        if with_lstm:
            self.lstm_1 = torch.nn.LSTM(self.config.hidden_size, self.config.hidden_size // 2, batch_first=True, bidirectional=True)
            # self.lstm_2 = torch.nn.LSTM(self.config.hidden_size, self.config.hidden_size // 2, batch_first=True, bidirectional=True)

    def forward(self, x, mask, lm=True):
        output = self.backbone(x, mask)
        all_seq_features = output[0]  # (bs, seq_len, dim)

        if self.with_lstm:
            all_seq_features, _ = self.lstm_1(all_seq_features)
            # all_seq_features, _ = self.lstm_2(all_seq_features)

        last_seq_feature = all_seq_features[:, 0] # (bs, dim)
        if lm and self.with_lm:
            return self.classifier(last_seq_feature), self.ranker(last_seq_feature), self.lm(last_seq_feature)
        else:
            return self.classifier(last_seq_feature), self.ranker(last_seq_feature)


class CodebertModel(nn.Module):

    def __init__(self, pretrained_path, dropout=0.2):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.backbone = AutoModel.from_pretrained(pretrained_path)

        out_features_num = self.config.dim

        self.mask_lm = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps),
            nn.Linear(self.config.hidden_size, self.config.vocab_size),
        )

        self.causal_lm = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps),
            nn.Linear(self.config.hidden_size, self.config.vocab_size),
        )

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
