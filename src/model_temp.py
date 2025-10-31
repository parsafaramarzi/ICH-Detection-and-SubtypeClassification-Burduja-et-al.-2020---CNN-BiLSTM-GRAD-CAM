import torch
import torch.nn as nn
import torchvision.models as models

# --- Stage 1: Slice-Level CNN (Feature Extraction) ---

class SliceLevelCNN(nn.Module):
    """
    CNN backbone based on ResNeXt-101 (32x8d) for slice-level feature extraction.
    It returns the feature vector for the LSTM and auxiliary slice-level logits.
    """
    # Feature dimension of the ResNeXt-101 global average pool output
    FEATURE_DIM = 2048 
    # 5 hemorrhage subtypes + 1 'Any Hemorrhage' class
    NUM_SLICE_CLASSES = 6 

    def __init__(self, pretrained=True):
        super().__init__()
        
        # 1. Load the ResNeXt-101 (32x8d) model
        self.backbone = models.resnext101_32x8d(
            weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # 2. Remove the standard classification head (to use the features)
        self.backbone.fc = nn.Identity() 

        # 3. Define the custom linear head for slice-level classification
        self.slice_classifier = nn.Linear(self.FEATURE_DIM, self.NUM_SLICE_CLASSES)

    def forward(self, x):
        # Implementation of the forward pass
        features = self.backbone(x)
        slice_logits = self.slice_classifier(features)
        
        # Returns 2048-D features for the LSTM and 6 slice-level predictions for auxiliary loss
        return features, slice_logits

# --- Stage 2: Combined CNN-LSTM Model ---

class CombinedICHModel(nn.Module):
    """
    The full two-stage model: CNN features fed into a Bidirectional LSTM 
    to aggregate information across the sequence of slices (scan-level).
    """
    # Required LSTM parameters from the paper
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 3
    NUM_SCAN_CLASSES = 6 # Same 6 classes for the final prediction
    
    def __init__(self, cnn_pretrained=True):
        super().__init__()
        
        # 1. Instantiate the Slice-Level CNN
        self.cnn_stage = SliceLevelCNN(pretrained=cnn_pretrained)

        # 2. Define the Bidirectional LSTM aggregator
        self.lstm = nn.LSTM(
            input_size=SliceLevelCNN.FEATURE_DIM,       # 2048 features per slice
            hidden_size=self.LSTM_HIDDEN_SIZE,          # 256
            num_layers=self.LSTM_NUM_LAYERS,            # 3 layers
            batch_first=True,                           # Input shape (B, S, F)
            bidirectional=True                          # Crucial for bi-LSTM
        )

        # 3. Define the final linear head for scan-level classification
        # Input size is 2 * hidden_size due to bidirectionality (2*256 = 512)
        self.scan_classifier = nn.Linear(
            self.LSTM_HIDDEN_SIZE * 2,
            self.NUM_SCAN_CLASSES
        )
        
    def forward(self, x):
        # --- Part 3: The forward pass implementation goes here ---
        
        # 1. Get Batch (B) and Sequence (S) dimensions
        B, S, C, H, W = x.shape
        
        # 2. Flatten Sequence for CNN processing: (B*S, C, H, W)
        x_flat = x.view(B * S, C, H, W)
        
        # 3. Run CNN (Stage 1)
        # features_flat: (B*S, 2048), slice_logits: (B*S, 6)
        features_flat, slice_logits = self.cnn_stage(x_flat)
        
        # 4. Restore Sequence structure for LSTM input: (B, S, 2048)
        features_seq = features_flat.view(B, S, -1) 
        
        # 5. Run LSTM (Stage 2)
        # out: (B, S, 2*Hidden_Size), _ is the hidden/cell state (h_n, c_n)
        out, _ = self.lstm(features_seq)
        
        # 6. Extract the final output for scan-level classification
        # We take the output of the last sequence element: (B, 2*Hidden_Size)
        final_seq_output = out[:, -1, :] 
        
        # 7. Final Classification
        # Maps (B, 512) -> (B, 6) for the scan-level logits
        scan_logits = self.scan_classifier(final_seq_output)
        
        # Returns the primary scan prediction and the auxiliary slice predictions
        return scan_logits, slice_logits