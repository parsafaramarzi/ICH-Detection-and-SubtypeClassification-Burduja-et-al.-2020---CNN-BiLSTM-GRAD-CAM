from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models

class SliceLevelCNN(nn.Module):

    FEATURE_DIM = 2048
    NUM_SLICE_CLASSES = 6

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnext101_32x8d(weights = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.slice_classifier = nn.Linear(self.FEATURE_DIM, self.NUM_SLICE_CLASSES)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.slice_classifier(features)
        return features, logits

class SequenceAggregatorBiLSTM(nn.Module):

    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 3
    NUM_SCAN_CLASSES = 6

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=SliceLevelCNN.FEATURE_DIM, hidden_size=self.LSTM_HIDDEN_SIZE, num_layers=self.LSTM_NUM_LAYERS, batch_first=True, bidirectional=True)
        self.scan_classifier = nn.Linear((self.LSTM_HIDDEN_SIZE * 2) + 6, self.NUM_SCAN_CLASSES)

    def forward(self, features_seq: torch.Tensor, cnn_logits_seq: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(features_seq)
        combined_features = torch.cat([lstm_out, cnn_logits_seq], dim=2)
        final_slice_logits = self.scan_classifier(combined_features)
        return final_slice_logits

class FullModel(nn.Module):

    def __init__(self, cnn_pretrained=True):
        super().__init__()
        self.slice_level_cnn = SliceLevelCNN(pretrained=cnn_pretrained)
        self.bilstm = SequenceAggregatorBiLSTM()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W)
        features_flat, cnn_logits_flat = self.slice_level_cnn(x_flat)
        features_seq = features_flat.reshape(B, S, -1)
        cnn_logits_seq = cnn_logits_flat.reshape(B, S, -1)
        final_logits = self.bilstm(features_seq, cnn_logits_seq)
        return final_logits, cnn_logits_seq