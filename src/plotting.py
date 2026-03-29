import os
import argparse
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
    # ==========================================
    # 1. Argument Parser for Filtering
    # ==========================================
    parser = argparse.ArgumentParser(description="Generate comparison plots from experiment CSV files.")

    # Example usage: --include VGG11 _BS_ --exclude PROTONET Cutout
    parser.add_argument('--include', nargs='+', default=[],
                        help='Words that MUST ALL be present in the filename (AND logic)')
    parser.add_argument('--include_any', nargs='+', default=[],
                        help='Words where at least ONE must be present in the filename (OR logic)')
    parser.add_argument('--exclude', nargs='+', default=[],
                        help='Words that MUST NOT be present in the filename (NOT logic)')

    args = parser.parse_args()

    # ==========================================
    # 2. Define directory paths
    # ==========================================
    experiments_dir = 'experiments'
    plots_dir = 'plots'

    # Create the plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)

    # 2. Get a list of all .csv files from the experiments directory
    if not os.path.exists(experiments_dir):
        print(f"[*] Error: Directory '{experiments_dir}' not found!")
        return

    # ==========================================
    # 3. Read and Filter CSV Files
    # ==========================================
    all_files = [f for f in os.listdir(experiments_dir) if f.endswith('.csv')]

    # Apply filters
    filtered_csv_files = []
    for file in all_files:
        # Check if ALL 'include' words are in the filename (AND logic)
        # If args.include is empty, all() naturally returns True
        has_all_includes = all(word in file for word in args.include)

        # Check if ANY 'include_any' words are in the filename (OR logic)
        # If args.include_any is empty, we treat it as True so it doesn't block other filters
        has_any_include_any = any(word in file for word in args.include_any) if args.include_any else True

        # Check if ANY 'exclude' word is in the filename (NOT logic)
        has_any_excludes = any(word in file for word in args.exclude)

        # Final check: Must pass AND filter, OR filter, and not trigger NOT filter
        if has_all_includes and has_any_include_any and not has_any_excludes:
            filtered_csv_files.append(file)

    if len(filtered_csv_files) < 2:
        print(f"[*] Found only {len(filtered_csv_files)} files matching your criteria.")
        print("[*] Error: You need at least 2 .csv files to create comparison pairs!")
        print(f"    Current files in pool: {filtered_csv_files}")
        return

    print(f"[*] Found {len(filtered_csv_files)} matching CSV files out of {len(all_files)} total.")
    if args.include:
        print(f"    - Included (ALL of): {args.include}")
    if args.include_any:
        print(f"    - Included (ANY of): {args.include_any}")
    if args.exclude:
        print(f"    - Excluded: {args.exclude}")

    # ==========================================
    # 4. Generate pairs and Plot
    # ==========================================
    pairs = list(combinations(filtered_csv_files, 2))
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