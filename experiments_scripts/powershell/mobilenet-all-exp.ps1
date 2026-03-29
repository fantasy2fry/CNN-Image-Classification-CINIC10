# Skrypt do pelnych eksperymentow puszczanych na noc dla MobileNet
Write-Host "[!] Uruchamianie maratonu eksperymentow dla MobileNet..." -ForegroundColor Yellow

$PYTHON_CMD = ".\.venv\Scripts\python.exe"
$EPOCHS = 30
$BATCH = 128
$BATCH_NOT_FROZEN = 64

# --- 1. TRANSFER LEARNING (ZAMROZONE WAGI): Tylko ostatnia warstwa ---
Write-Host "`n>>> [1/8] Transfer Learning: Adam (LR=0.001)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model mobilenet --epochs $EPOCHS --batch_size $BATCH --freeze_features --optimizer adam --lr 0.001

Write-Host "`n>>> [2/8] Transfer Learning: SGD (LR=0.01)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model mobilenet --epochs $EPOCHS --batch_size $BATCH --freeze_features --optimizer sgd --lr 0.01


# --- 2. PELNY FINE-TUNING (ODMROZONE WAGI): Cala siec sie uczy ---
Write-Host "`n>>> [3/8] Fine-Tuning: Adam (LR=0.0001)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model mobilenet --epochs $EPOCHS --batch_size $BATCH_NOT_FROZEN --optimizer adam --lr 0.0001

Write-Host "`n>>> [4/8] Fine-Tuning: SGD (LR=0.001)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model mobilenet --epochs $EPOCHS --batch_size $BATCH_NOT_FROZEN --optimizer sgd --lr 0.001


# --- 3. REGULARYZACJA W TRANSFER LEARNINGU ---
Write-Host "`n>>> [5/8] Transfer Learning + Weight Decay (Adam, L2=1e-4)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model mobilenet --epochs $EPOCHS --batch_size $BATCH --freeze_features --optimizer adam --lr 0.001 --weight_decay 0.0001

Write-Host "`n>>> [6/8] Transfer Learning + Cutout (Adam)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model mobilenet --epochs $EPOCHS --batch_size $BATCH --freeze_features --optimizer adam --lr 0.001 --use_cutout


# --- 4. ZAAWANSOWANY FINE-TUNING (KOMBO) ---
Write-Host "`n>>> [7/8] Fine-Tuning + Weight Decay + Cutout (Adam, LR=0.0001)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model mobilenet --epochs $EPOCHS --batch_size $BATCH_NOT_FROZEN --optimizer adam --lr 0.0001 --weight_decay 0.0001 --use_cutout

Write-Host "`n>>> [8/8] Fine-Tuning + Weight Decay + Cutout (SGD, LR=0.001)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model mobilenet --epochs $EPOCHS --batch_size $BATCH_NOT_FROZEN --optimizer sgd --lr 0.001 --weight_decay 0.0001 --use_cutout

Write-Host "`n[!] Wszystkie 8 eksperymentow dla MobileNet zakonczone sukcesem!" -ForegroundColor Green
