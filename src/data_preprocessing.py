import os
import numpy as np
import torch
import nibabel as nib
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from typing import Union

def determine_optimal_num_slices(
    data_source: Union[str, pd.DataFrame], 
    patient_id_col: str, 
    slice_count_col: str
) -> int:
 
    if isinstance(data_source, str):
        try:
            master_df = pd.read_csv(data_source)
            print(f"Successfully loaded DataFrame from path: {data_source}")
        except FileNotFoundError:
            print(f"Error: CSV file not found at path: {data_source}")
            return 0
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return 0
    elif isinstance(data_source, pd.DataFrame):
        master_df = data_source
    else:
        print("Error: data_source must be a file path (str) or a Pandas DataFrame.")
        return 0


    if master_df.empty:
        print("Warning: Input DataFrame is empty. Returning 0 slices.")
        return 0

    if slice_count_col not in master_df.columns or patient_id_col not in master_df.columns:
        print(f"Error: Missing required columns in DataFrame. Need '{patient_id_col}' and '{slice_count_col}'.")
        return 0
        
    patient_slice_counts = master_df.groupby(patient_id_col)[slice_count_col].max()

    if patient_slice_counts.empty:
        print(f"Error: Could not extract slice counts after grouping by '{patient_id_col}'.")
        return 0
        
    min_slices = patient_slice_counts.min()

    print(f"Found {len(patient_slice_counts)} unique patient scans.")
    print(f"Maximum total slice count observed: {patient_slice_counts.max()}")
    print(f"The optimal (minimum) total number of slices to use for consistent sampling is: {min_slices}")

    return int(min_slices)

def display_slice(volume, labels, index, denormalize=True):
    N = volume.shape[0]
    if not (1 <= index <= N):
        print(f"Index {index} out of bounds (1 to {N}).")
        return
    

    img_data = volume[index - 1].cpu().numpy()
    if denormalize:
        MEAN = np.array([0.1738, 0.1433, 0.1970]).reshape(3, 1, 1)
        STD = np.array([0.3161, 0.2850, 0.3111]).reshape(3, 1, 1)
        denormalized = (img_data * STD) + MEAN
        denormalized = np.clip(denormalized, 0, 1)
        slice_img = denormalized.transpose(1, 2, 0)
    else:
        slice_img = img_data.transpose(1, 2, 0)

    label_vec = labels[index - 1].cpu().numpy()
    names = ['IVH', 'IPH', 'SAH', 'EDH', 'SDH', 'ANY']
    
    positive = [name for name, val in zip(names, label_vec) if val == 1.0]
    
    if positive:
        label_text = ", ".join(positive)
        c = 'red'
    else:
        label_text = "No Hemorrhage"
        c = 'green'

    plt.figure(figsize=(6, 7))
    plt.imshow(slice_img)
    plt.title(f"Slice {index} of {N}")
    plt.xlabel(f"Label: {label_text}", color=c, fontsize=12)
    plt.tight_layout()
    plt.show()

def apply_window(img, wc, ww):
    lower = wc - ww / 2
    upper = wc + ww / 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    return img

def Normalize(img):
    mean = np.array([0.1738, 0.1433, 0.1970])
    std = np.array([0.3161, 0.2850, 0.3111])
    img = (img - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    return img

def Augmentations(augmentation_list):
    transforms = []
    
    # 1. Horizontal Flipping
    if 'flip' in augmentation_list:
        transforms.append(A.HorizontalFlip(p=1.0)) 
    
    # 2. Shifting (Translation)
    if 'shift' in augmentation_list:
        transforms.append(A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},scale=(1.0, 1.0),rotate=(0, 0),interpolation=cv2.INTER_LINEAR,p=1.0,fill=-255,border_mode=cv2.BORDER_CONSTANT))
        
    # 3. Rotation
    if 'rotate' in augmentation_list:
        transforms.append(A.Rotate(limit=[-180, 180],interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_CONSTANT,p=1.0,fill=-255))
        
    # 4. Scaling (Zoom In/Out)
    if 'scale' in augmentation_list:
        transforms.append(A.Affine(translate_percent={'x': (0, 0), 'y': (0, 0)},scale=(0.85, 1.15),rotate=(0, 0),keep_ratio=True,border_mode=cv2.BORDER_CONSTANT,fill=-255,interpolation=cv2.INTER_LINEAR,p=1.0))
        
    # 5. Brightness Adjustment
    if 'brightness' in augmentation_list:
        transforms.append(A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.2],contrast_limit=[0, 0],p=1.0,brightness_by_max=True,ensure_safe_range=True))

    return A.Compose(transforms)

def data_load(patient_id, target_size, label_dir = "data/", data_dir = "data/ct_scans", num_slices=16, transform_list=None):

    img_nifti = nib.load(os.path.join(data_dir, f"{patient_id:03d}.nii"))
    volume = img_nifti.get_fdata().astype(np.float32)

    labels_df = pd.read_csv(os.path.join(label_dir, "hemorrhage_diagnosis_raw_ct.csv"))
    patient_labels = labels_df[labels_df["PatientNumber"] == patient_id].sort_values("SliceNumber")    
    labels_matrix_raw = patient_labels.iloc[:, 2:8].values.astype(np.float32) 
    labels_matrix_raw[:, 5] = 1.0 - labels_matrix_raw[:, 5]

    Slice_Counts = volume.shape[2]
    label_rows_count = patient_labels.shape[0]
    if Slice_Counts != label_rows_count:
        raise ValueError(f"Warning: Slice count ({Slice_Counts}) and label rows count ({label_rows_count}) do not match.")
    
    augmentations = Augmentations(transform_list) if transform_list is not None else None
    indices = np.linspace(0, Slice_Counts - 1, num_slices, dtype=int)
    labels_matrix_raw = labels_matrix_raw[indices, :]
    processed_slices = []
    for i in indices:
        img_slice = volume[:, :, i].T
            
        # Apply HU windowing
        brain = apply_window(img_slice, wc=40, ww=80)
        subdural = apply_window(img_slice, wc=80, ww=200)
        soft = apply_window(img_slice, wc=40, ww=380)
        stacked = np.stack([brain, subdural, soft], axis=-1)

        # Resize
        stacked = cv2.resize(stacked, target_size)

        # Normalization
        stacked = Normalize(stacked)

        # Augmentation
        if transform_list is not None:
            augmented = augmentations(image=stacked)
            stacked = augmented['image']

        stacked = stacked.transpose(2, 0, 1)
        processed_slices.append(stacked)

    volume_tensor = torch.from_numpy(np.stack(processed_slices)).float()
    labels_tensor = torch.from_numpy(labels_matrix_raw.copy()).float()
    print("-"*80)
    print(f"Patient ID: {patient_id}, total number of slices and labels rows from {Slice_Counts},{label_rows_count} to {volume_tensor.shape[0]},{labels_tensor.shape[0]}")
    print(f"CT_Scan Shape: {volume_tensor.shape}")
    print(f"Labels Shape: {labels_tensor.shape}")
    print("-"*80)


    return volume_tensor, labels_tensor

if __name__ == "__main__":
    PATIENT_ID = 49
    TARGET_SIZE = (256, 256)
    LABEL_DIR = "data/"
    DATA_DIR = "data/ct_scans"
    NUM_SLICES = 16
    TRANSFORM_LIST = ["rotate", "flip", "scale", "shift", "brightness"]
    DISPLAY_SLICE_INDEX = 7

    volume_tensor, labels_tensor = data_load(PATIENT_ID, TARGET_SIZE, LABEL_DIR, DATA_DIR, num_slices=NUM_SLICES, transform_list=TRANSFORM_LIST)
    display_slice(volume_tensor, labels_tensor, DISPLAY_SLICE_INDEX, denormalize=False)