import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


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
    parser = argparse.ArgumentParser(description="Generate a combined Validation Accuracy plot at specific epochs.")

    parser.add_argument('--include', nargs='+', default=[],
                        help='Words that MUST ALL be present in the filename (AND logic)')
    parser.add_argument('--include_any', nargs='+', default=[],
                        help='Words where at least ONE must be present in the filename (OR logic)')
    parser.add_argument('--exclude', nargs='+', default=[],
                        help='Words that MUST NOT be present in the filename (NOT logic)')
    parser.add_argument('--outname', type=str, default='Combined_Val_Accuracy.png',
                        help='Name of the output plot file (default: Combined_Val_Accuracy.png)')

    args = parser.parse_args()

    # ==========================================
    # 2. Define directory paths
    # ==========================================
    experiments_dir = 'experiments'
    plots_dir = 'plots'

    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(experiments_dir):
        print(f"[*] Error: Directory '{experiments_dir}' not found!")
        return

    # ==========================================
    # 3. Read and Filter CSV Files
    # ==========================================
    all_files = [f for f in os.listdir(experiments_dir) if f.endswith('.csv')]

    # Ignoruj pliki leaderboardów, jeśli są w folderze
    if 'experiments_leaderboard.csv' in all_files:
        all_files.remove('experiments_leaderboard.csv')

    filtered_csv_files = []
    for file in all_files:
        has_all_includes = all(word in file for word in args.include)
        has_any_include_any = any(word in file for word in args.include_any) if args.include_any else True
        has_any_excludes = any(word in file for word in args.exclude)

        if has_all_includes and has_any_include_any and not has_any_excludes:
            filtered_csv_files.append(file)

    if not filtered_csv_files:
        print(f"[*] Found 0 files matching your criteria.")
        return

    print(f"[*] Found {len(filtered_csv_files)} matching CSV files out of {len(all_files)} total.")

    # ==========================================
    # 4. Generate Single Combined Plot
    # ==========================================
    # Definiujemy epoki, które nas interesują
    target_epochs = [1, 5, 10, 15, 20, 25, 30]

    # Tworzymy jeden duży wykres
    plt.figure(figsize=(14, 8))

    for file in filtered_csv_files:
        path = os.path.join(experiments_dir, file)
        df = pd.read_csv(path)
        name = clean_label(file)

        # Filtrujemy DataFrame, aby zostawić tylko interesujące nas epoki
        df_filtered = df[df['Epoch'].isin(target_epochs)].sort_values('Epoch')

        if not df_filtered.empty:
            # Rysujemy linię z punktami (marker='o') dla danego modelu
            plt.plot(df_filtered['Epoch'], df_filtered['Accuracy'],
                     marker='o', markersize=8, linewidth=2, label=name)
        else:
            print(f"[!] Warning: No matching epochs found in {file}")

    # ==========================================
    # Plot configuration
    # ==========================================
    plt.title('Validation Accuracy Comparison', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)

    # Wymuszamy na osi X wyświetlanie tylko tych konkretnych epok
    plt.xticks(target_epochs)

    plt.grid(True, linestyle=':', alpha=0.7)

    # Legenda pod wykresem
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=1)

    # Dostosowanie marginesów
    plt.tight_layout()
    # Robimy miejsce na dole na potencjalnie bardzo długą legendę
    # Im więcej plików, tym więcej miejsca trzeba na dole
    margin_bottom = min(0.5, 0.15 + (len(filtered_csv_files) * 0.03))
    plt.subplots_adjust(bottom=margin_bottom)

    # ==========================================
    # Save the plot
    # ==========================================
    outpath = os.path.join(plots_dir, args.outname)
    plt.savefig(outpath, dpi=150)
    plt.close()

    print(f"\n[*] Done! Combined plot saved to: {outpath}")


if __name__ == '__main__':
    main()