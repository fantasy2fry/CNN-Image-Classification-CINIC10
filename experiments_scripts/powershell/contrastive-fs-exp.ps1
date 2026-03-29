# MobileNet Contrastive Few-Shot Experiments (frozen backbone)
Write-Host "[!] Starting Contrastive Few-Shot marathon for MobileNet..." -ForegroundColor Yellow

$PYTHON_CMD = ".\.venv\Scripts\python.exe"
$EPOCHS = 30
$BATCH = 128

# --- 1. BASELINE: 10-shot, default margin ---
Write-Host "`n>>> [1/6] Baseline 10-shot: Adam, LR=0.001, Margin=1.0" -ForegroundColor Cyan
&$PYTHON_CMD src/train_fs.py --model mobilenet --epochs $EPOCHS --batch_size $BATCH --samples_per_class 10 --freeze_features --optimizer adam --lr 0.001 --margin 1.0

# --- 2. MARGIN TUNING: stronger class separation in embedding space ---
Write-Host "`n>>> [2/6] High Margin 10-shot: Adam, LR=0.001, Margin=2.0" -ForegroundColor Cyan
&$PYTHON_CMD src/train_fs.py --model mobilenet --epochs $EPOCHS --batch_size $BATCH --samples_per_class 10 --freeze_features --optimizer adam --lr 0.001 --margin 2.0

# --- 3. L2 REGULARIZATION: prevents projection head overfitting on 100 images ---
Write-Host "`n>>> [3/6] Weight Decay 10-shot: Adam, LR=0.001, WD=1e-4, Margin=1.0" -ForegroundColor Cyan
&$PYTHON_CMD src/train_fs.py --model mobilenet --epochs $EPOCHS --batch_size $BATCH --samples_per_class 10 --freeze_features --optimizer adam --lr 0.001 --margin 1.0 --weight_decay 0.0001

# --- 4. CUTOUT: stronger view diversity boosts contrastive learning signal ---
Write-Host "`n>>> [4/6] Cutout 10-shot: Adam, LR=0.001, Cutout ON, Margin=1.0" -ForegroundColor Cyan
&$PYTHON_CMD src/train_fs.py --model mobilenet --epochs $EPOCHS --batch_size $BATCH --samples_per_class 10 --freeze_features --optimizer adam --lr 0.001 --margin 1.0 --use_cutout

# --- 5. MORE DATA: 50-shot to show how k-NN accuracy scales with support set size ---
Write-Host "`n>>> [5/6] Baseline 50-shot: Adam, LR=0.001, Margin=1.0" -ForegroundColor Cyan
&$PYTHON_CMD src/train_fs.py --model mobilenet --epochs $EPOCHS --batch_size $BATCH --samples_per_class 50 --freeze_features --optimizer adam --lr 0.001 --margin 1.0

# --- 6. ABLATION: no augmentation exposes how critical view diversity is in contrastive ---
Write-Host "`n>>> [6/6] No Augmentation 10-shot: Adam, LR=0.001, no crop/flip/rotation" -ForegroundColor Cyan
&$PYTHON_CMD src/train_fs.py --model mobilenet --epochs $EPOCHS --batch_size $BATCH --samples_per_class 10 --freeze_features --optimizer adam --lr 0.001 --margin 1.0 --disable_crop --disable_flip --disable_rotation

Write-Host "`n[!] All 6 Contrastive Few-Shot experiments finished!" -ForegroundColor Green