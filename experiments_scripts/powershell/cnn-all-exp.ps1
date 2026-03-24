# Skrypt do pelnych eksperymentow puszczanych na noc dla Custom CNN
Write-Host "[!] Uruchamianie maratonu eksperymentow dla Custom CNN..." -ForegroundColor Yellow

$PYTHON_CMD = ".\.venv\Scripts\python.exe"
$EPOCHS = 30
$BATCH = 128

# --- 1. BAZA: Rozne optymalizatory i Learning Rate (bez regularyzacji) ---
Write-Host "`n>>> [1/8] Baza Adam (LR=0.001)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model cnn --epochs $EPOCHS --batch_size $BATCH --optimizer adam --lr 0.001 --dropout 0.0

Write-Host "`n>>> [2/8] Baza Adam z mniejszym LR (LR=0.0001)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model cnn --epochs $EPOCHS --batch_size $BATCH --optimizer adam --lr 0.0001 --dropout 0.0

Write-Host "`n>>> [3/8] Baza SGD (LR=0.01)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model cnn --epochs $EPOCHS --batch_size $BATCH --optimizer sgd --lr 0.01 --dropout 0.0


# --- 2. DROPOUT: Sprawdzamy wplyw sredniego dropoutu (0.5) ---
Write-Host "`n>>> [4/8] Adam + Dropout (0.5)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model cnn --epochs $EPOCHS --batch_size $BATCH --optimizer adam --lr 0.001 --dropout 0.5

Write-Host "`n>>> [5/8] SGD + Dropout (0.5)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model cnn --epochs $EPOCHS --batch_size $BATCH --optimizer sgd --lr 0.01 --dropout 0.5


# --- 3. ZAAWANSOWANA REGULARYZACJA: Weight Decay i Cutout ---
Write-Host "`n>>> [6/8] Adam + Weight Decay (L2 = 1e-4)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model cnn --epochs $EPOCHS --batch_size $BATCH --optimizer adam --lr 0.001 --dropout 0.0 --weight_decay 0.0001

Write-Host "`n>>> [7/8] SGD + Cutout (Data Augmentation)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model cnn --epochs $EPOCHS --batch_size $BATCH --optimizer sgd --lr 0.01 --dropout 0.0 --use_cutout


# --- 4. KOMBO: Wszystkie techniki na raz ---
Write-Host "`n>>> [8/8] ULTIMATE RUN: Adam + Cutout + Dropout (0.3) + Weight Decay" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model cnn --epochs $EPOCHS --batch_size $BATCH --optimizer adam --lr 0.001 --dropout 0.3 --weight_decay 0.0001 --use_cutout

Write-Host "`n[!] Wszystkie 8 eksperymentow dla Custom CNN zakonczone sukcesem!" -ForegroundColor Green
