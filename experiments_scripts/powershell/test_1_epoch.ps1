# Skrypt testowy - sprawdzenie czy wszystkie eksperymenty "dzialaja" bez bledow przez 1 epoke
Write-Host "[!] Uruchamianie pierwszych eksperymentów do weryfikacji sprzętu..." -ForegroundColor Yellow

$PYTHON_CMD = ".\.venv\Scripts\python.exe"
$EPOCHS = 1
$BATCH = 128

# 1. Custom CNN Baseline (najważniejszy)
Write-Host ">>> Uruchamianie EXPERIMENT 1: Custom CNN Baseline (1 Epoch)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model cnn --epochs $EPOCHS --batch_size $BATCH --lr 0.001

# 2. Custom CNN + Dropout
Write-Host ">>> Uruchamianie EXPERIMENT 2: Custom CNN + Dropout=0.5 (1 Epoch)" -ForegroundColor Cyan
&$PYTHON_CMD src/train.py --model cnn --epochs $EPOCHS --batch_size $BATCH --dropout 0.5

Write-Host "[!] Podstawowe testy (1 epoka) zakonczone pomyslnie!" -ForegroundColor Green
