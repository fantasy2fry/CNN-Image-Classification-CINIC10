import os
import pandas as pd
import argparse


def main():
    # ==========================================
    # 1. Argument Parser for Filtering & Saving
    # ==========================================
    parser = argparse.ArgumentParser(description="Generate a leaderboard from experiment CSV files.")

    parser.add_argument('--include', nargs='+', default=[],
                        help='Words that MUST be present in the filename (e.g., VGG11 _BS_)')
    parser.add_argument('--exclude', nargs='+', default=[],
                        help='Words that MUST NOT be present in the filename (e.g., PROTONET Cutout)')
    parser.add_argument('--filename', type=str, default='experiments_leaderboard.csv',
                        help='Name of the output CSV file for the leaderboard')
    parser.add_argument('--latex', action='store_true',
                        help='Generate a LaTeX table (.tex) alongside the CSV')

    args = parser.parse_args()

    experiments_dir = 'experiments'

    if not os.path.exists(experiments_dir):
        print(f"[*] Error: Directory '{experiments_dir}' not found!")
        return

    all_csv_files = [f for f in os.listdir(experiments_dir) if f.endswith('.csv')]

    output_filename = args.filename if args.filename.endswith('.csv') else f"{args.filename}.csv"

    for file_to_skip in [output_filename, 'experiments_leaderboard.csv']:
        if file_to_skip in all_csv_files:
            all_csv_files.remove(file_to_skip)

    if not all_csv_files:
        print("[*] No experiment CSV files found.")
        return

    # ==========================================
    # 2. Filter the CSV files
    # ==========================================
    filtered_csv_files = []
    for file in all_csv_files:
        has_all_includes = all(word in file for word in args.include)
        has_any_excludes = any(word in file for word in args.exclude)

        if has_all_includes and not has_any_excludes:
            filtered_csv_files.append(file)

    if not filtered_csv_files:
        print(f"[*] Found 0 files matching your criteria.")
        return

    print(f"[*] Scanning {len(filtered_csv_files)} experiment files...\n")

    results = []

    # ==========================================
    # 3. Process the files
    # ==========================================
    for file in filtered_csv_files:
        file_path = os.path.join(experiments_dir, file)
        try:
            df = pd.read_csv(file_path)

            best_val_acc = df['Accuracy'].max()
            best_val_loss = df['Loss'].min()
            best_train_acc = df['Train_Accuracy'].max()
            best_epoch = df.loc[df['Accuracy'].idxmax(), 'Epoch']

            model_name = file.replace('.csv', '')
            overfit_warning = "Yes" if (best_train_acc - best_val_acc > 0.15) else "No"

            results.append({
                'Model Configuration': model_name,
                'Best Val Acc': best_val_acc,
                'Best Val Loss': best_val_loss,
                'Epoch': int(best_epoch),
                'Overfit?': overfit_warning
            })
        except Exception as e:
            print(f"[*] Error reading {file}: {e}")

    if not results:
        return

    # Create a DataFrame and sort by Best Validation Accuracy
    leaderboard_df = pd.DataFrame(results)
    leaderboard_df = leaderboard_df.sort_values(by='Best Val Acc', ascending=False).reset_index(drop=True)

    # Start index from 1 for ranking
    leaderboard_df.index = leaderboard_df.index + 1
    leaderboard_df.index.name = 'Rank'

    # ==========================================
    # 4. Display & Save CSV
    # ==========================================
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.width', 200)

    print(leaderboard_df.to_string())
    print("\n==========================================================================================")

    save_path = os.path.join(experiments_dir, output_filename)
    #leaderboard_df.to_csv(save_path, index=True)  # Zapisujemy index (Rank) do CSV
    print(f"[*] Leaderboard CSV saved to '{save_path}'")

    # ==========================================
    # 5. Generate LaTeX Table
    # ==========================================
    if args.latex:
        latex_df = leaderboard_df.copy()

        latex_df['Model Configuration'] = latex_df['Model Configuration'].str.replace('_', r'\_', regex=False)

        latex_filename = output_filename.replace('.csv', '.tex')
        latex_path = os.path.join(experiments_dir, latex_filename)

        try:
            latex_str = latex_df.style.format(precision=4).to_latex()
        except AttributeError:
            latex_str = latex_df.to_latex(float_format="%.4f")

        with open(latex_path, 'w') as f:
            f.write(latex_str)

        print(f"[*] LaTeX source code saved to '{latex_path}'")


if __name__ == '__main__':
    main()