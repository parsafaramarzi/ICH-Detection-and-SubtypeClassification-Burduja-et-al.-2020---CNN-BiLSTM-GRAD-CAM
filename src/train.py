import os, torch, torch.nn.functional as F, torch.optim as optim, numpy as np, pandas as pd, random
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from data_preprocessing import data_load
from model import FullModel 
from utils import append_to_csv_log, ensure_directory_exists, get_next_batch_log_filename, get_next_epoch_log_filename

class HemorrhageDataset(Dataset):
    def __init__(self, patient_ids, target_size, label_dir, data_dir, num_slices, transform_list):
        self.patient_ids = patient_ids
        self.target_size = target_size
        self.label_dir = label_dir
        self.data_dir = data_dir
        self.num_slices = num_slices
        self.transform_list = transform_list

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        volume, labels = data_load(
            patient_id=patient_id, 
            target_size=self.target_size, 
            label_dir=self.label_dir, 
            data_dir=self.data_dir, 
            num_slices=self.num_slices,
            transform_list=self.transform_list
        )
        return volume, labels

def weighted_bce_loss(logits, targets, device):
    WEIGHTS = torch.tensor([2., 2., 2., 2., 2., 4.], dtype=torch.float32).to(device)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (loss * WEIGHTS).mean()

def save_checkpoint(epoch, model, optimizer, directory):
    os.makedirs(directory, exist_ok=True)
    torch.save({'Epoch': epoch, 'Model': model.state_dict(), 'Optimizer': optimizer.state_dict()}, os.path.join(directory, f"checkpoint_epoch_{epoch}.pth"))
    print(f"Checkpoint saved for epoch {epoch}")

def load_checkpoint(checkpoint, model, optimizer):
    if not os.path.isfile(checkpoint):
        print(f"No checkpoint found at {checkpoint}. Starting from epoch 1.")
        return 1
    checkpoint_data = torch.load(checkpoint)
    model.load_state_dict(checkpoint_data['Model'])
    optimizer.load_state_dict(checkpoint_data['Optimizer'])
    print(f"Loaded checkpoint from {checkpoint}")
    return checkpoint_data['Epoch'] + 1

def train_one_epoch(epoch, model, loader, optimizer, device, aux_loss_weight, batch_log_filepath):
    model.train(); print(f"\n--- Epoch {epoch} Training Started ---")
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        bilstm_labels_preds, cnn_labels_preds = model(images)

        cnn_labels_preds_flat = cnn_labels_preds.view(-1, 6)
        scan_labels_preds_flat = bilstm_labels_preds.view(-1, 6)
        labels_flat = labels.view(-1, 6)

        primary_loss = weighted_bce_loss(scan_labels_preds_flat, labels_flat, device)
        auxiliary_loss = weighted_bce_loss(cnn_labels_preds_flat, labels_flat, device)
        total_loss = primary_loss + aux_loss_weight * auxiliary_loss
        total_loss.backward()
        optimizer.step()

        print("#"*80)
        print(f"  Batch {i+1}/{len(loader)} | Total Loss: {total_loss.item():.4f} | Primary Loss (BiLSTM): {primary_loss.item():.4f} | Auxiliary Loss (CNN): {auxiliary_loss.item():.4f}")
        print("#"*80)
        Log_Header = [
            "epoch",
            "batch_num",
            "total_batches",
            "total_loss",
            "primary_loss",
            "auxiliary_loss",
            "learning_rate"
        ]
        log_data = [
            epoch,
            i + 1,
            len(loader),
            round(total_loss.item(), 4),
            round(primary_loss.item(), 4),
            round(auxiliary_loss.item(), 4),
            round(optimizer.param_groups[0]['lr'], 8)
        ]
        append_to_csv_log(batch_log_filepath, Log_Header, log_data)

def test_one_epoch(epoch, model, loader, device, aux_loss_weight, epoch_log_filepath):
    model.eval(); print(f"\n--- Epoch {epoch} Testing ---")
    
    # For Patient-Level Macro AUC (Any hemorrhage present)
    all_preds_patient, all_targets_patient = [], [] 
    
    # For Slice/Class-Level Metrics (6 classes)
    all_preds_multi_logits, all_targets_multi = [], [] 
    
    total_loss_accum = 0.0
    primary_loss_accum = 0.0
    auxiliary_loss_accum = 0.0
    
    HEMORRHAGE_CLASSES = [
        "Intraparenchymal", "Subdural", "Epidural", 
        "Intraventricular", "Subarachnoid", "Any"
    ]
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            s_log, cnn_labels_preds = model(images) 

            # --- Loss Calculation ---
            cnn_labels_preds_flat = cnn_labels_preds.view(-1, 6)
            scan_labels_preds_flat = s_log.view(-1, 6)
            labels_flat = labels.view(-1, 6)

            primary_loss = weighted_bce_loss(scan_labels_preds_flat, labels_flat, device)
            auxiliary_loss = weighted_bce_loss(cnn_labels_preds_flat, labels_flat, device)
            total_loss = primary_loss + aux_loss_weight * auxiliary_loss

            total_loss_accum += total_loss.item()
            primary_loss_accum += primary_loss.item()
            auxiliary_loss_accum += auxiliary_loss.item()

            # --- Patient-Level AUC (Any Hemorrhage) ---
            
            # Step 1: Get the max sigmoid score per class across all slices (Shape: B x 6)
            max_scores_per_class, _ = torch.max(torch.sigmoid(s_log), dim=1)
            
            # Step 2: Get the single maximum score across ALL classes for the patient (Shape: B)
            # This represents the highest probability of ANY hemorrhage in the scan.
            p_preds, _ = torch.max(max_scores_per_class, dim=1)
            
            # Patient target is True if any slice has any label set (Shape: B)
            p_targets = labels.any(dim=1).any(dim=1).float() 

            all_preds_patient.append(p_preds.cpu().numpy())
            all_targets_patient.append(p_targets.cpu().numpy())
            
            # --- Slice/Class-Level Metrics Data Collection ---
            all_preds_multi_logits.append(scan_labels_preds_flat.cpu().numpy())
            all_targets_multi.append(labels_flat.cpu().numpy())

    num_batches = len(loader)
    avg_total_loss = total_loss_accum / num_batches
    avg_primary_loss = primary_loss_accum / num_batches
    avg_auxiliary_loss = auxiliary_loss_accum / num_batches
    
    # Combine collected data
    targets_multi = np.concatenate(all_targets_multi, axis=0)
    preds_logits = np.concatenate(all_preds_multi_logits, axis=0)
    preds_scores = torch.sigmoid(torch.from_numpy(preds_logits)).numpy()
    
    # --- 1. Patient-Level Macro AUC ---
    patient_auc = roc_auc_score(np.concatenate(all_targets_patient), np.concatenate(all_preds_patient), average='macro')

    # --- 2. Class-Specific AUC (on the slice level) ---
    class_auc = []
    
    # FIX: Iterate through each class and only calculate AUC if both classes (0 and 1) are present
    for i in range(targets_multi.shape[1]):
        y_true_col = targets_multi[:, i]
        y_score_col = preds_scores[:, i]
        
        # Check if both 0 and 1 classes are present in the ground truth
        if len(np.unique(y_true_col)) == 2:
            # Calculate AUC if it's defined
            auc = roc_auc_score(y_true_col, y_score_col)
            class_auc.append(auc)
        else:
            # Assign a neutral score (0.5) if only one class is present
            class_auc.append(0.5)
            print(f"Warning: Only one class present in y_true for class {HEMORRHAGE_CLASSES[i]}. Setting AUC to 0.5.")
            
    auc_dict = {f"auc_{cls.lower()}": class_auc[i] for i, cls in enumerate(HEMORRHAGE_CLASSES)}
    
    # --- 3. Threshold Metrics (0.5 threshold) ---
    # Convert scores to binary predictions (0 or 1)
    threshold = 0.5
    binary_preds = (preds_scores >= threshold).astype(int)
    
    # Accuracy (Micro-Averaged)
    acc = accuracy_score(targets_multi, binary_preds)

    # Precision, Recall, F1-Score (Micro-Averaged, often preferred for multi-label)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_multi, binary_preds, average='micro', zero_division=0
    )
    
    # --- 4. Confusion Matrix Components (Micro-Averaged Counts) ---
    # We sum TP, TN, FP, FN across all classes and all slices
    tn, fp, fn, tp = confusion_matrix(targets_multi.ravel(), binary_preds.ravel(), labels=[0, 1]).ravel()

    print(f"Test Patient Macro AUC: {patient_auc:.4f}")
    print(f"Test Micro Accuracy: {acc:.4f} | Micro P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}")
    print(f"Test Class AUCs: {auc_dict}")
    
    LOG_HEADER = [
    "epoch",
    "patient_macro_auc",
    "avg_total_loss",
    "avg_primary_loss",
    "avg_auxiliary_loss",
    "micro_accuracy",
    "micro_precision",
    "micro_recall",
    "micro_f1_score",
    "tp_count",
    "tn_count",
    "fp_count",
    "fn_count",
    ] + list(auc_dict.keys())
    
    log_data = [
    epoch,
    round(patient_auc, 4),
    round(avg_total_loss, 4),
    round(avg_primary_loss, 4), 
    round(avg_auxiliary_loss, 4),
    round(acc, 4),
    round(precision, 4),
    round(recall, 4),
    round(f1, 4),
    int(tp),
    int(tn),
    int(fp),
    int(fn),
    ] + [round(auc_val, 4) for auc_val in auc_dict.values()]
    
    append_to_csv_log(epoch_log_filepath, LOG_HEADER, log_data)
    return patient_auc

if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Starting training script on {device}")
    
    DATA_ROOT = "data/"
    MASTER_CSV = os.path.join(DATA_ROOT, "hemorrhage_diagnosis_raw_ct.csv")
    CT_SCANS_DIR = os.path.join(DATA_ROOT, "ct_scans")
    BATCH_SIZE = 2
    IMG_SIZE = 256
    NUM_SLICES = 16
    EPOCHS = 10
    LR = 1e-4
    AUXILIARY_LOSS_WEIGHT = 0.5
    CHECKPOINT_DIR = "checkpoints" 
    RESUME_CHECKPOINT = None
    LOG_DIR = "outputs/logs"

    
    TRANSFORM_TRAIN = ["rotate", "flip", "scale", "shift", "brightness"]
    TRANSFORM_TEST = [] 

    print("Loading and splitting master labels...")
    full_df = pd.read_csv(MASTER_CSV)
    unique_ids = full_df['PatientNumber'].unique() 
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=SEED)
    print(f"Training Patients: {len(train_ids)}, Test Patients: {len(test_ids)}")
    
    print("Initializing Datasets...")
    train_ds = HemorrhageDataset(train_ids, (IMG_SIZE, IMG_SIZE), DATA_ROOT, CT_SCANS_DIR, NUM_SLICES, TRANSFORM_TRAIN)
    test_ds = HemorrhageDataset(test_ids, (IMG_SIZE, IMG_SIZE), DATA_ROOT, CT_SCANS_DIR, NUM_SLICES, TRANSFORM_TEST)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, 1, shuffle=False, num_workers=2)

    print("Initializing FullModel...")
    model = FullModel(cnn_pretrained=True)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    print("Model ready.")

    start_epoch = 1
    if RESUME_CHECKPOINT: start_epoch = load_checkpoint(RESUME_CHECKPOINT, model, optimizer)

    ensure_directory_exists(LOG_DIR)
    
    TRAIN_BATCH_LOG_FILEPATH = get_next_batch_log_filename(LOG_DIR)
    TEST_EPOCH_LOG_FILEPATH = get_next_epoch_log_filename(LOG_DIR)
    print(f"Batch logs will be saved to: {TRAIN_BATCH_LOG_FILEPATH}")
    print(f"Epoch summary logs will be saved to: {TEST_EPOCH_LOG_FILEPATH}")

    for epoch in range(start_epoch, EPOCHS + 1):
        train_one_epoch(epoch, model, train_loader, optimizer, device, AUXILIARY_LOSS_WEIGHT, TRAIN_BATCH_LOG_FILEPATH)
        test_one_epoch(epoch, model, test_loader, device, AUXILIARY_LOSS_WEIGHT, TEST_EPOCH_LOG_FILEPATH) 
        save_checkpoint(epoch, model, optimizer, CHECKPOINT_DIR)
    print(f"\nTraining complete after {EPOCHS} epochs.")