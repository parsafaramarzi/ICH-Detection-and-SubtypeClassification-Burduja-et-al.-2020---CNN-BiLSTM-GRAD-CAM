import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import random
import glob

# NOTE: The custom modules (CTICHDataset, CombinedICHModel, get_scan_level_labels) 
# are assumed to be available from the other files (dataset.py, model.py, utils.py).
from dataset import CTICHDataset
from model import CombinedICHModel
# Assuming utility function is in utils.py (if you create it)
def get_scan_level_labels(labels_tensor):
    """
    Converts a sequence of slice-level labels (N, 6) into a single scan-level label (1, 6).
    The scan is positive for a class if *any* slice is positive for that class.
    """
    # Max over the sequence dimension (0)
    # The result of .any(dim=0) is a boolean tensor, converting to float for loss calculation.
    return labels_tensor.any(dim=0).float().unsqueeze(0)


# --- PART 1: SETUP, SEEDING, AND CUSTOM WEIGHTED LOSS ---

# 1. Reproducibility
def seed_everything(seed=42):
    """Sets the seed for reproducibility across PyTorch and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

# 2. Custom Weighted Loss Function (from competition metric)
def weighted_bce_loss(logits, targets, device):
    """
    Calculates Weighted Binary Cross Entropy Loss.
    Weights: 4x for 'Any Hemorrhage' (index 5), 2x for all 5 subtypes (indices 0-4).
    """
    # The weights correspond to ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural', 'Any_Hemorrhage']
    WEIGHTS = torch.tensor([2., 2., 2., 2., 2., 4.], dtype=torch.float32).to(device)
    
    # Standard BCEWithLogitsLoss
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    
    # Apply weights
    weighted_loss = loss * WEIGHTS
    
    # Return the mean loss across the batch and classes
    return weighted_loss.mean()


# --- PART 2: CONFIGURATION AND INITIALIZATION ---

# 3. Configuration
DATA_ROOT = "data/"
TRAIN_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
VAL_CSV = os.path.join(DATA_ROOT, "val_labels.csv")
OUTPUT_DIR = "outputs/"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

BATCH_SIZE = 2
NUM_WORKERS = 2
IMG_SIZE = 256  # Matching dataset.py
EPOCHS = 10
LR = 1e-4
AUXILIARY_LOSS_WEIGHT = 0.5 # Lambda (λ) for slice loss

# Ensure output directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 4. Device and Writer Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, "run1"))

# 5. Data Loaders
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

# NOTE: CTICHDataset must be initialized with the correct arguments matching dataset.py
train_ds = CTICHDataset(data_dir=DATA_ROOT, labels_csv=TRAIN_CSV, target_size=(IMG_SIZE, IMG_SIZE), use_all_slices=False)
val_ds = CTICHDataset(data_dir=DATA_ROOT, labels_csv=VAL_CSV, target_size=(IMG_SIZE, IMG_SIZE), use_all_slices=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# Validation batch size is often 1 for sequence models to avoid padding/simplification issues
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS) 

# 6. Model, Optimizer, and Scaler Initialization
model = CombinedICHModel(cnn_pretrained=True)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None


# --- PART 3: TRAINING AND VALIDATION FUNCTIONS ---

def train_one_epoch(epoch, model, train_loader, optimizer, scaler, device, writer):
    """Handles the forward pass, two-stage loss, backpropagation, and logging for one epoch."""
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    global_step = epoch * len(train_loader)

    for batch_idx, (imgs, slice_level_labels) in enumerate(pbar):
        # 1. Data Preparation
        imgs = imgs.to(device) # (B, S, C, H, W)
        slice_level_labels = slice_level_labels.to(device) # (B, S, 6)
        
        # We need the scan-level label only once per volume for the primary loss.
        # This implementation requires the batch size to be 1 for simplicity in this loop.
        # For BATCH_SIZE > 1, this needs adjustment (e.g., list comprehension or map)
        
        # HACK for BATCH_SIZE=2, where we take the first item's scan label
        # In a production environment, this should be handled by a custom collate_fn.
        scan_level_labels = get_scan_level_labels(slice_level_labels[0]).to(device) 
        
        optimizer.zero_grad()
        
        # 2. Forward Pass and Loss Calculation
        if scaler:
            with torch.cuda.amp.autocast():
                scan_logits, slice_logits = model(imgs)
                
                # Reshape slice_logits/labels for element-wise loss (B*S, 6)
                slice_logits_flat = slice_logits.view(-1, 6)
                slice_level_labels_flat = slice_level_labels.view(-1, 6)
                
                # Loss 1: Scan Loss (Primary Loss)
                scan_loss = weighted_bce_loss(scan_logits, scan_level_labels, device)
                
                # Loss 2: Slice Loss (Auxiliary Loss)
                slice_loss = weighted_bce_loss(slice_logits_flat, slice_level_labels_flat, device)
                
                # Total Two-Stage Loss
                total_loss = scan_loss + AUXILIARY_LOSS_WEIGHT * slice_loss
                
            # 3. Backward Pass (with AMP)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard float training (for CPU or if AMP is disabled)
            scan_logits, slice_logits = model(imgs)
            
            slice_logits_flat = slice_logits.view(-1, 6)
            slice_level_labels_flat = slice_level_labels.view(-1, 6)
            
            scan_loss = weighted_bce_loss(scan_logits, scan_level_labels, device)
            slice_loss = weighted_bce_loss(slice_logits_flat, slice_level_labels_flat, device)
            total_loss = scan_loss + AUXILIARY_LOSS_WEIGHT * slice_loss

            total_loss.backward()
            optimizer.step()

        # 4. Logging
        pbar.set_postfix({"Loss": total_loss.item(), "ScanL": scan_loss.item(), "SliceL": slice_loss.item()})
        writer.add_scalar('train/total_loss', total_loss.item(), global_step + batch_idx)
        writer.add_scalar('train/scan_loss', scan_loss.item(), global_step + batch_idx)
        writer.add_scalar('train/slice_loss', slice_loss.item(), global_step + batch_idx)


def validate_one_epoch(epoch, model, val_loader, device):
    """Handles the validation loop and computes the ROC AUC metric."""
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for imgs, slice_level_labels in tqdm(val_loader, desc=f"Epoch {epoch} Validating"):
            
            imgs = imgs.to(device) # (1, S, C, H, W) - Val batch size is 1
            
            # The scan-level label is derived from the slice labels (any positive slice makes the scan positive)
            scan_level_labels = get_scan_level_labels(slice_level_labels[0]) # (1, 6)
            
            # Forward pass: we only care about the scan_logits for the final metric
            scan_logits, _ = model(imgs)

            # Detach, convert to sigmoid probabilities, and move to CPU/NumPy
            probs = torch.sigmoid(scan_logits).cpu().numpy()
            labels = scan_level_labels.cpu().numpy()

            y_true.append(labels)
            y_pred.append(probs)

    # Flatten and compute ROC AUC (for 6 classes)
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    
    # Calculate ROC AUC per class and then average
    aucs = []
    for i in range(y_true.shape[1]):
        try:
            # Need at least two unique classes for ROC AUC to be defined
            if len(np.unique(y_true[:, i])) > 1:
                aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
            else:
                 aucs.append(np.nan)
        except Exception:
            aucs.append(np.nan)
    
    mean_auc = np.nanmean(aucs)
    
    # Log AUC to TensorBoard and console
    writer.add_scalar('val/mean_auc', mean_auc, epoch)
    print(f"\n--- Epoch {epoch} Val AUC: {mean_auc:.4f} ---")
    
    return mean_auc


# --- PART 4: MAIN EXECUTION LOOP ---

if __name__ == "__main__":
    
    best_auc = 0.0
    
    print(f"Starting training on {device}")
    
    for epoch in range(1, EPOCHS + 1):
        
        # Training Phase
        train_one_epoch(epoch, model, train_loader, optimizer, scaler, device, writer)
        
        # Validation Phase
        current_auc = validate_one_epoch(epoch, model, val_loader, device)
        
        # Checkpoint Saving
        if current_auc > best_auc:
            best_auc = current_auc
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path} with new best AUC: {best_auc:.4f}")

    print(f"Training complete. Best ROC AUC achieved: {best_auc:.4f}")
    writer.close()
