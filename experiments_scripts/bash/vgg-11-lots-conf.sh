#!/bin/bash

# Zatrzymuje skrypt, jeśli któryś z modeli wyrzuci błąd
set -e

EPOCHS=30
SCRIPT="python -m src.train --model vgg11 --epochs $EPOCHS"

echo "=========================================================================="
echo " Rozpoczynam rygorystyczny Grid Search & Ablation Study dla modelu VGG-11 "
echo " Całkowita liczba eksperymentów: 20 | Epoki: $EPOCHS"
echo "=========================================================================="

# =====================================================================
# FAZA 1: ABLATION STUDY - BADANIE AUGMENTACJI
# Cel: Sprawdzić, które zniekształcenia obrazu faktycznie pomagają,
# a które może nawet przeszkadzają VGG-11 (Optimizer: Adam 1e-3, bez regularyzacji).
# =====================================================================
echo -e "\n[FAZA 1/5] BAZA AUGMENTACJI"

echo "---> [1/20] BAZA: Brak augmentacji (Czysty overfitting?)"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.0 --disable_crop --disable_flip --disable_rotation

echo "---> [2/20] ABLATION: Tylko Horizontal Flip (Najbezpieczniejsza augmentacja)"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.0 --disable_crop --disable_rotation

echo "---> [3/20] ABLATION: Crop + Flip (Brak rotacji - częsty standard)"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.0 --disable_rotation

echo "---> [4/20] STANDARD: Pełna podstawowa augmentacja (Crop + Flip + Rot)"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.0

# Od tego momentu wszystkie modele używają pełnej, podstawowej augmentacji!


# =====================================================================
# FAZA 2: LEARNING RATE SWEEP (Z pełną augmentacją)
# Cel: Znaleźć idealny rozmiar kroku dla obydwu optymalizatorów.
# =====================================================================
echo -e "\n[FAZA 2/5] LEARNING RATE SWEEP"

# ADAM
echo "---> [5/20] LR TEST (Adam): Wysoki LR = 0.005"
$SCRIPT --optimizer adam --lr 0.005 --dropout 0.0

echo "---> [6/20] LR TEST (Adam): Standard LR = 0.001 (Wykonane w 4/20, pomijamy)"

echo "---> [7/20] LR TEST (Adam): Niski LR = 0.0001 (Może wolniej, ale dokładniej?)"
$SCRIPT --optimizer adam --lr 0.0001 --dropout 0.0

echo "---> [8/20] LR TEST (Adam): Bardzo niski LR = 0.00001"
$SCRIPT --optimizer adam --lr 0.00001 --dropout 0.0

# SGD (Zawsze potrzebuje większego LR na start niż Adam)
echo "---> [9/20] LR TEST (SGD): Bardzo wysoki LR = 0.1"
$SCRIPT --optimizer sgd --lr 0.1 --dropout 0.0

echo "---> [10/20] LR TEST (SGD): Wysoki LR = 0.01"
$SCRIPT --optimizer sgd --lr 0.01 --dropout 0.0


# =====================================================================
# FAZA 3: BATCH SIZE DYNAMICS
# Cel: Sprawdzić wpływ "szumu" w gradiencie (Adam LR 1e-3).
# =====================================================================
echo -e "\n[FAZA 3/5] BATCH SIZE"

echo "---> [11/20] BATCH TEST: Mały Batch = 64 (Więcej aktualizacji, więcej szumu)"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.0 --batch_size 64

echo "---> [12/20] BATCH TEST: Duży Batch = 256 (Gładszy gradient, stabilniej)"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.0 --batch_size 256


# =====================================================================
# FAZA 4: WALKA Z OVERFITTINGIEM (REGULARYZACJA)
# Cel: Badamy optymalny poziom "utrudnień" dla sieci.
# =====================================================================
echo -e "\n[FAZA 4/5] REGULARYZACJA (DROPOUT, L2, CUTOUT)"

# DROPOUT SWEEP
echo "---> [13/20] DROPOUT: Lekki = 0.3"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.3

echo "---> [14/20] DROPOUT: Średni = 0.5 (Standard)"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.5

echo "---> [15/20] DROPOUT: Agresywny = 0.7 (Czy model sobie poradzi?)"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.7

# WEIGHT DECAY (L2) SWEEP (Kary za zbyt duże wagi)
echo "---> [16/20] L2 PENALTY: Mocna = 1e-3"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.0 --weight_decay 0.001

echo "---> [17/20] L2 PENALTY: Delikatna = 1e-4"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.0 --weight_decay 0.0001

# CUTOUT
echo "---> [18/20] CUTOUT: Test agresywnej augmentacji"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.0 --use_cutout


# =====================================================================
# FAZA 5: FRANKENSTEIN (Złożenie najlepszych praktyk)
# Cel: Połączenie kilku metod na raz. Często dają efekt synergii.
# =====================================================================
echo -e "\n[FAZA 5/5] MULTI-REGULARIZATION (ULTIMATE RUNS)"

echo "---> [19/20] KOMBO 1: Adam + Cutout + Dropout (0.3) + Delikatne L2 (1e-4)"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.3 --weight_decay 0.0001 --use_cutout

echo "---> [20/20] KOMBO 2: SGD (Wolny, ale solidny) LR=0.01 + Dropout (0.5) + Cutout + L2"
$SCRIPT --optimizer sgd --lr 0.01 --dropout 0.5 --weight_decay 0.0001 --use_cutout

echo -e "\n=========================================================================="
echo " Wszystkie 20 eksperymentów zakończone pomyślnie!"
echo " Skrypt wygenerował potężną paczkę wiedzy. Odpal plot_experiments.py!"
echo "=========================================================================="