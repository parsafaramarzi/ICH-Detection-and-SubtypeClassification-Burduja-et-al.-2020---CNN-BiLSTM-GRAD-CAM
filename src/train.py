import os, torch, torch.nn.functional as F, torch.optim as optim, numpy as np, pandas as pd, random
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score
from data_preprocessing import data_load
from model import FullModel 

# --- Wrapper Dataset Class (REQUIRED) ---
class HemorrhageDataset(Dataset):
    """
    Wraps the data_load function to create a PyTorch Dataset.
    Manages the list of patient IDs and loads data on-the-fly.
    """
    def __init__(self, patient_ids, target_size, label_dir, data_dir, transform_list):
        self.patient_ids = patient_ids
        self.target_size = target_size
        self.label_dir = label_dir
        self.data_dir = data_dir
        self.transform_list = transform_list

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        # Get the specific patient ID for this index
        patient_id = self.patient_ids[idx]
        
        # Load the preprocessed volume and labels for this patient
        # data_load returns: volume (S, 3, H, W), labels (S, 6)
        volume, labels = data_load(
            patient_id=patient_id, 
            target_size=self.target_size, 
            label_dir=self.label_dir, 
            data_dir=self.data_dir, 
            transform_list=self.transform_list
        )
        return volume, labels

def weighted_bce_loss(logits, targets, device):
    WEIGHTS = torch.tensor([2., 2., 2., 2., 2., 4.], dtype=torch.float32).to(device)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (loss * WEIGHTS).mean()

def save_checkpoint(e, m, o, d):
    os.makedirs(d, exist_ok=True); torch.save({'e': e, 'm': m.state_dict(), 'o': o.state_dict()}, os.path.join(d, f"checkpoint_epoch_{e}.pth")); print(f"Checkpoint saved for epoch {e}")

def load_checkpoint(p, m, o):
    if not os.path.isfile(p): print(f"No checkpoint found at {p}. Starting from epoch 1."); return 1
    c = torch.load(p); m.load_state_dict(c['m']); o.load_state_dict(c['o']); print(f"Loaded checkpoint from {p}"); return c['e'] + 1

def train_one_epoch(e, m, l, o, d, w):
    m.train(); print(f"\n--- Epoch {e} Training Started ---")
    for i, (imgs, lbls) in enumerate(l):
        imgs, lbls = imgs.to(d), lbls.to(d); o.zero_grad()
        s_log, c_log = m(imgs)
        s_flat, c_flat, l_flat = s_log.view(-1, 6), c_log.view(-1, 6), lbls.view(-1, 6)
        p_loss = weighted_bce_loss(s_flat, l_flat, d); a_loss = weighted_bce_loss(c_flat, l_flat, d); t_loss = p_loss + w * a_loss
        t_loss.backward(); o.step()
        if (i + 1) % 100 == 0: print(f"  Batch {i+1}/{len(l)} | Loss: {t_loss.item():.4f}")

def test_one_epoch(e, m, l, d):
    m.eval(); print(f"\n--- Epoch {e} Testing ---"); all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, lbls in l:
            imgs, lbls = imgs.to(d), lbls.to(d); s_log, _ = m(imgs)
            p_preds, _ = torch.max(torch.sigmoid(s_log), dim=1); p_targets = lbls.any(dim=1).float()
            all_preds.append(p_preds.cpu().numpy()); all_targets.append(p_targets.cpu().numpy())
    auc = roc_auc_score(np.concatenate(all_targets), np.concatenate(all_preds), average='macro')
    print(f"Test Macro AUC: {auc:.4f}"); return auc

if __name__ == "__main__":
    SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print("CUDA support removed. Starting training script on CPU."); device = torch.device("cpu")
    
    # Configuration
    DATA_ROOT = "data/"
    MASTER_CSV = os.path.join(DATA_ROOT, "hemorrhage_diagnosis_raw_ct.csv")
    CT_SCANS_DIR = os.path.join(DATA_ROOT, "ct_scans")
    BATCH_SIZE, IMG_SIZE, EPOCHS, LR, AUXILIARY_LOSS_WEIGHT = 4, 256, 10, 1e-4, 0.5
    CHECKPOINT_DIR, RESUME_CHECKPOINT = "checkpoints", None 
    
    # Define Augmentations
    TRANSFORM_TRAIN = ["rotate", "flip", "scale", "shift", "brightness"]
    TRANSFORM_TEST = [] 

    print("Loading and splitting master labels...")
    full_df = pd.read_csv(MASTER_CSV); unique_ids = full_df['PatientNumber'].unique() 
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=SEED)
    print(f"Training Patients: {len(train_ids)}, Test Patients: {len(test_ids)}")
    
    print("Initializing Datasets...")
    # Initialize the Wrapper Dataset with the list of Patient IDs
    train_ds = HemorrhageDataset(train_ids, (IMG_SIZE, IMG_SIZE), DATA_ROOT, CT_SCANS_DIR, TRANSFORM_TRAIN)
    test_ds = HemorrhageDataset(test_ids, (IMG_SIZE, IMG_SIZE), DATA_ROOT, CT_SCANS_DIR, TRANSFORM_TEST)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, 1, shuffle=False, num_workers=2)

    print("Initializing FullModel..."); model = FullModel(cnn_pretrained=True); model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR); print("Model ready.")

    start_epoch = 1
    if RESUME_CHECKPOINT: start_epoch = load_checkpoint(RESUME_CHECKPOINT, model, optimizer)

    for epoch in range(start_epoch, EPOCHS + 1):
        train_one_epoch(epoch, model, train_loader, optimizer, device, AUXILIARY_LOSS_WEIGHT)
        test_one_epoch(epoch, model, test_loader, device)
        save_checkpoint(epoch, model, optimizer, CHECKPOINT_DIR)
        
    print(f"\nTraining complete after {EPOCHS} epochs.")