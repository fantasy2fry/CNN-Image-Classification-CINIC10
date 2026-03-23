import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations


def clean_label(filename):
    """
    Cleans the filename to be used as a readable label in the plot legend.
    Removes the '.csv' extension.
    """
    return filename.replace('.csv', '')


def main():
    # 1. Define directory paths
    experiments_dir = 'experiments'
    plots_dir = 'plots'

    # Create the plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)

    # 2. Get a list of all .csv files from the experiments directory
    if not os.path.exists(experiments_dir):
        print(f"[*] Error: Directory '{experiments_dir}' not found!")
        return

    csv_files = [f for f in os.listdir(experiments_dir) if f.endswith('.csv')]

    if len(csv_files) < 2:
        print("[*] Error: You need at least 2 .csv files in the 'experiments' folder to create comparison pairs!")
        return

    print(f"[*] Found {len(csv_files)} CSV files. Generating pairs...")

    # 3. Generate all possible pairs (combinations of 2)
    pairs = list(combinations(csv_files, 2))
    print(f"[*] Total plots to generate: {len(pairs)}\n")

    # 4. Loop through each pair and generate the comparison plot
    for i, (file1, file2) in enumerate(pairs, 1):
        path1 = os.path.join(experiments_dir, file1)
        path2 = os.path.join(experiments_dir, file2)

        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)

        # Clean names for the legend
        name1 = clean_label(file1)
        name2 = clean_label(file2)

        # Create a wide figure with two subplots side-by-side (Loss and Accuracy)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # ==========================================
        # PLOT 1: CROSS ENTROPY LOSS
        # ==========================================
        # Model 1 (Blue)
        ax1.plot(df1['Epoch'], df1['Train_Loss'], color='blue', linestyle='--', alpha=0.6, label=f'Train: {name1}')
        ax1.plot(df1['Epoch'], df1['Loss'], color='blue', linestyle='-', linewidth=2, label=f'Valid: {name1}')

        # Model 2 (Orange/Red)
        ax1.plot(df2['Epoch'], df2['Train_Loss'], color='orangered', linestyle='--', alpha=0.6, label=f'Train: {name2}')
        ax1.plot(df2['Epoch'], df2['Loss'], color='orangered', linestyle='-', linewidth=2, label=f'Valid: {name2}')

        ax1.set_title('Cross Entropy Loss Comparison', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, linestyle=':', alpha=0.7)

        # Place the legend below the plot to avoid overlapping with data lines
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=1)

        # ==========================================
        # PLOT 2: ACCURACY
        # ==========================================
        # Model 1 (Blue)
        ax2.plot(df1['Epoch'], df1['Train_Accuracy'], color='blue', linestyle='--', alpha=0.6, label=f'Train: {name1}')
        ax2.plot(df1['Epoch'], df1['Accuracy'], color='blue', linestyle='-', linewidth=2, label=f'Valid: {name1}')

        # Model 2 (Orange/Red)
        ax2.plot(df2['Epoch'], df2['Train_Accuracy'], color='orangered', linestyle='--', alpha=0.6,
                 label=f'Train: {name2}')
        ax2.plot(df2['Epoch'], df2['Accuracy'], color='orangered', linestyle='-', linewidth=2, label=f'Valid: {name2}')

        ax2.set_title('Accuracy Comparison', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, linestyle=':', alpha=0.7)

        # Place the legend below the plot
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=1)

        # ==========================================
        # SAVE THE PLOT
        # ==========================================
        # Adjust layout so the bottom legend doesn't get cut off
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.35)

        # Output filename: Model1_VS_Model2.png
        plot_filename = f"{name1}_VS_{name2}.png"
        plot_path = os.path.join(plots_dir, plot_filename)

        plt.savefig(plot_path, dpi=150)

        # Close the figure to free up RAM (crucial when generating many plots!)
        plt.close()

        print(f"[{i}/{len(pairs)}] Saved: {plot_filename}")

    print("\n[*] Done! All comparison plots have been saved in the 'plots/' directory.")


if __name__ == '__main__':
    main()