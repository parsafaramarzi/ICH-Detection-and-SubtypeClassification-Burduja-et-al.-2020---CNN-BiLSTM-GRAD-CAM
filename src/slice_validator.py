from dataset import CTICHDataset

# --- Configuration (Must match your dataset.py setup) ---
# Ensure these paths are correct for your local setup to run successfully.
DATA_DIR = "data/ct_scans"
LABELS_CSV = "data/hemorrhage_diagnosis_raw_ct.csv"

# 1. Initialize the Dataset
# This loads the CSV and sets up the patient list.
dataset = CTICHDataset(
    data_dir=DATA_DIR,
    labels_csv=LABELS_CSV,
    target_size=(256, 256),
    use_all_slices=True
)

# 2. Retrieve the first data sample (Patient 0)
# This calls __getitem__(0) and returns the full volume and all slice labels.
volume_tensor, labels_tensor = dataset[0]

# 3. Print the results (Format and Sample Values)
print(f"--- Quick Data Inspection ---")
print(f"Volume Tensor Shape (N, C, H, W): {volume_tensor.shape}")
print(f"Labels Tensor Shape (N, 6): {labels_tensor.shape}")
print(f"Volume Tensor Data Type: {volume_tensor.dtype}")
print(f"Labels Tensor Data Type: {labels_tensor.dtype}")

# Print the values of the first 5 pixels in the first channel of the very first slice
print("\nFirst 5 pixel values (Slice 0, Channel 0):")
print(volume_tensor[0, 0, 0, :5].numpy())

# Print the label vector for the first two slices (N_slice, 6_labels)
print("\nLabels for the first 2 slices (5 subtypes + Any Hemorrhage):")
print(labels_tensor[:2].numpy())
