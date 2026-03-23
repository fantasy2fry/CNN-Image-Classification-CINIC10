# Skrypt do pelnych eksperymentow puszczanych na noc
Write-Host "[!] Uruchamianie maratonu eksperymentow na noc..." -ForegroundColor Yellow

$PYTHON_CMD = ".\.venv\Scripts\python.exe"
$EPOCHS_CNN = 20
$EPOCHS_MOBILENET = 10
$BATCH = 128

# 1. Custom CNN Baseline
Write-Host "`n>>> EXPERIMENT 1: Custom CNN Baseline ($EPOCHS_CNN Epok)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model cnn --epochs $EPOCHS_CNN --batch_size $BATCH --lr 0.001

# 2. Custom CNN + Dropout
Write-Host "`n>>> EXPERIMENT 2: Custom CNN + Dropout=0.5 ($EPOCHS_CNN Epok)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model cnn --epochs $EPOCHS_CNN --batch_size $BATCH --dropout 0.5

# 3. Custom CNN + L2 (Weight Decay)
Write-Host "`n>>> EXPERIMENT 3: Custom CNN + L2/Weight Decay=0.01 ($EPOCHS_CNN Epok)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model cnn --epochs $EPOCHS_CNN --batch_size $BATCH --weight_decay 0.01

# 4. Custom CNN + Cutout (Data Augmentation)
Write-Host "`n>>> EXPERIMENT 4: Custom CNN + Cutout ($EPOCHS_CNN Epok)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model cnn --epochs $EPOCHS_CNN --batch_size $BATCH --use_cutout

# 5. MobileNet (Fine-Tuning)
Write-Host "`n>>> EXPERIMENT 5: MobileNet Finetuning ($EPOCHS_MOBILENET Epok)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model mobilenet --epochs $EPOCHS_MOBILENET --batch_size $BATCH --lr 0.001

Write-Host "`n[!] Wszystkie epickie nocne eksperymenty zakonczone sukcesem!" -ForegroundColor Green
