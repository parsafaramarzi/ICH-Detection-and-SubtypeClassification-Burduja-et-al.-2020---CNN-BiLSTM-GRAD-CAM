import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import cv2
import pandas as pd
from scipy import ndimage

def apply_window(img, wc, ww):
    lower = wc - ww / 2
    upper = wc + ww / 2
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)
    return img


class CTICHDataset(Dataset):

    def __init__(self, data_dir, labels_csv, transform=None,target_size=(256,256), use_all_slices = True, preserve_aspect_ratio=True):
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.unique_ids = self.labels_df['PatientNumber'].unique()
        self.transform = transform
        self.target_size = target_size
        self.use_all_slices = use_all_slices
        self.label_cols = ['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural']

    def __len__(self):
        return len(self.unique_ids)
    
    def __getitem__(self, index):
        subject_id = self.unique_ids[index]
        file_path = os.path.join(self.data_dir, f"{subject_id:03d}.nii")
        img_nifti = nib.load(file_path)
        volume = img_nifti.get_fdata().astype(np.float32)
        total_slices = volume.shape[2]
        patient_labels = self.labels_df[self.labels_df["PatientNumber"] == subject_id].sort_values("SliceNumber")

        if self.use_all_slices:
            indices = list(range(total_slices))
        else:
            indices = np.linspace(0, total_slices - 1, 16, dtype=int)

        processed_slices = []
        labels = []
        for i in indices:
            img_slice = volume[:, :, i].T
            
            # Apply HU windowing
            brain = apply_window(img_slice, wc=40, ww=80)
            subdural = apply_window(img_slice, wc=80, ww=200)
            soft = apply_window(img_slice, wc=40, ww=380)
            stacked = np.stack([brain, subdural, soft], axis=-1)

            stacked = cv2.resize(stacked, self.target_size)

            # Normalization
            mean = np.array([0.1738, 0.1433, 0.1970])
            std = np.array([0.3161, 0.2850, 0.3111])
            stacked = (stacked - mean) / std

            # Augmentation
            if self.transform:
                augmented = self.transform(image=stacked)
                stacked = augmented["image"]
            stacked = stacked.transpose(2, 0, 1)
            processed_slices.append(stacked)

            # Getting the multi-labels
            row = patient_labels.iloc[i]
            subtype_labels = row[self.label_cols].values.astype(float)
            any_label = 1 - row['No_Hemorrhage']
            slice_label = np.concatenate([subtype_labels, [any_label]])
            labels.append(slice_label)

        # Getting the outputs ready in the correct format
        volume_tensor = torch.from_numpy(np.stack(processed_slices)).float()
        labels_tensor = torch.from_numpy(np.stack(labels)).float()

        return volume_tensor, labels_tensor