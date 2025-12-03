import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the path to your CSV file. This assumes it's in the same directory, 
# but you might need to adjust this path based on where you run the script.
MASTER_CSV_PATH = 'data/hemorrhage_diagnosis_raw_ct.csv'

def visualize_label_distribution(csv_path: str):
    """
    Loads the master CSV, calculates the count of each hemorrhage subtype 
    (including No_Hemorrhage) at the slice level, and visualizes the results.

    Args:
        csv_path: The file path to the hemorrhage diagnosis CSV.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at path: {csv_path}")
        print("Please ensure the CSV file is correctly located or update the MASTER_CSV_PATH.")
        return

    # 1. Identify all relevant label columns
    # Based on the CSV structure:
    HEMORRHAGE_SUBTYPES = [
        'Intraventricular',
        'Intraparenchymal',
        'Subarachnoid',
        'Epidural',
        'Subdural',
        'No_Hemorrhage'
    ]
    
    # Filter columns to only include the ones we care about and check if they exist
    label_columns = [col for col in HEMORRHAGE_SUBTYPES if col in df.columns]
    
    if not label_columns:
        print("Error: Could not find necessary label columns in the CSV.")
        return

    print(f"Analyzing label distribution across {len(df)} slices...")

    # 2. Calculate the total count for each label
    # We sum the '1's in each column to get the total number of slices belonging to that class.
    distribution = df[label_columns].sum()
    
    # Optional: Calculate total positive slices to better highlight 'No_Hemorrhage'
    # A slice is positive if any of the subtype columns (excluding No_Hemorrhage) is 1
    subtype_cols = [c for c in label_columns if c != 'No_Hemorrhage']
    total_positive_slices = df[subtype_cols].max(axis=1).sum()
    total_slices = len(df)
    
    # Create the data structure for plotting
    plot_data = distribution.copy()
    
    # 3. Visualization using Matplotlib
    plt.figure(figsize=(12, 7))
    
    # Define colors, highlighting the "No_Hemorrhage" class
    colors = ['#FFC300', '#FF5733', '#C70039', '#900C3F', '#581845', '#1F618D']
    # Use a specific color for 'No_Hemorrhage' (the last one)
    num_classes = len(plot_data)
    bar_colors = colors[:num_classes-1] + ['#2874A6'] 
    
    # Create the bar chart
    bars = plt.bar(plot_data.index, plot_data.values, color=bar_colors)

    # Add labels and title
    plt.xlabel("Hemorrhage Subtype (Slice Level)", fontsize=14)
    plt.ylabel("Number of Slices (Log Scale)", fontsize=14)
    plt.title("Distribution of Slice-Level Hemorrhage Labels (Class Imbalance)", fontsize=16, weight='bold')
    
    # Use a logarithmic scale for the y-axis to better visualize small counts
    plt.yscale('log')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add data labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        # Display the actual count (not log value)
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.02, 
                 f'{int(yval):,}', 
                 ha='center', va='bottom', fontsize=10)

    # Add descriptive text box about the scale
    plt.annotate(
        "Note: Y-axis uses a Logarithmic Scale to show small positive counts clearly. "
        f"Total Slices: {total_slices:,}\n"
        f"Slices with Hemorrhage (Any Type): {total_positive_slices:,}",
        xy=(0.02, 0.98), xycoords='axes fraction',
        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7, ec="gray")
    )
    
    plt.xticks(rotation=15, ha='right', fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # You may need to change MASTER_CSV_PATH here if you run the script separately.
    visualize_label_distribution(MASTER_CSV_PATH)