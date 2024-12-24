import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss
from transformers import RobertaModel

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.manual_dense = nn.Linear(config['feature_size'], config['hidden_size'])
        self.fc1 = nn.Linear(config['hidden_size'] + config['hidden_size'], config['hidden_size'])
        self.fc2 = nn.Linear(config['hidden_size'], 1)
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.relu = nn.ReLU()

    def forward(self, features, manual_features=None, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS]) [batch_size, hidden_size]
        y = manual_features.float()  # [batch_size, feature_size]
        y = self.manual_dense(y)
        y = torch.tanh(y)

        x = torch.cat((x, y), dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class FineTuneModel(nn.Module):
    def __init__(self, config, tokenizer, args):
        super(FineTuneModel, self).__init__()
        self.encoder = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(self, input_ids, attention_mask, manual_features=None, labels=None, output_attentions=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions)

        last_layer_attn_weights = outputs.attentions[self.config['num_hidden_layers'] - 1][:, :, 0].detach() if output_attentions else None

        logits = self.classifier(outputs[0], manual_features)
        prob = torch.sigmoid(logits)

        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(prob, labels.unsqueeze(1).float())
            return loss, prob, last_layer_attn_weights
        else:
            return prob
