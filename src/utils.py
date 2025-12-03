import os
import csv

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def append_to_csv_log(filepath, header, data):
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)

def get_next_batch_log_filename(log_dir, base_name="training_batch_log"):
    i = 1
    while True:
        filename = f"{base_name}_{i:05d}.csv"
        filepath = os.path.join(log_dir, filename)
        
        if not os.path.exists(filepath):
            return filepath
        
        i += 1

def get_next_epoch_log_filename(log_dir, base_name="training_epoch_log"):
    i = 1
    while True:
        filename = f"{base_name}_{i:05d}.csv"
        filepath = os.path.join(log_dir, filename)
        
        if not os.path.exists(filepath):
            return filepath
        
        i += 1