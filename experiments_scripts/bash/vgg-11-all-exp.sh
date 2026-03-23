#!/bin/bash

# Zatrzymuje skrypt, jeśli któryś z modeli wyrzuci błąd
set -e

# Zmienne ogólne
EPOCHS=30
# Podmień "src.train_macos" na dokładną nazwę Twojego pliku Python
SCRIPT="python -m src.train_macos"

echo "=================================================="
echo " Uruchamiam nocną serię eksperymentów dla VGG-11! "
echo " Liczba epok dla każdego modelu: $EPOCHS"
echo "=================================================="

# --- 1. BAZA: Różne optymalizatory i Learning Rate (bez regularyzacji) ---
echo -e "\n---> [1/8] Baza Adam (LR=0.001)"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.0 --epochs $EPOCHS

echo -e "\n---> [2/8] Baza Adam z mniejszym LR (LR=0.0001)"
$SCRIPT --optimizer adam --lr 0.0001 --dropout 0.0 --epochs $EPOCHS

echo -e "\n---> [3/8] Baza SGD (SGD zazwyczaj wymaga wyższego LR, np. 0.01)"
$SCRIPT --optimizer sgd --lr 0.01 --dropout 0.0 --epochs $EPOCHS


# --- 2. DROPOUT: Sprawdzamy wpływ średniego dropoutu (0.5) ---
echo -e "\n---> [4/8] Adam + Dropout (0.5)"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.5 --epochs $EPOCHS

echo -e "\n---> [5/8] SGD + Dropout (0.5)"
$SCRIPT --optimizer sgd --lr 0.01 --dropout 0.5 --epochs $EPOCHS


# --- 3. ZAAWANSOWANA REGULARYZACJA: Weight Decay i Cutout ---
echo -e "\n---> [6/8] Adam + Weight Decay (L2 = 1e-4)"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.0 --weight_decay 0.0001 --epochs $EPOCHS

echo -e "\n---> [7/8] SGD + Cutout (Data Augmentation)"
$SCRIPT --optimizer sgd --lr 0.01 --dropout 0.0 --use_cutout --epochs $EPOCHS


# --- 4. KOMBO: Wszystkie techniki na raz ---
echo -e "\n---> [8/8] ULTIMATE RUN: Adam + Cutout + Dropout (0.3) + Weight Decay"
$SCRIPT --optimizer adam --lr 0.001 --dropout 0.3 --weight_decay 0.0001 --use_cutout --epochs $EPOCHS

echo -e "\n=================================================="
echo " Wszystkie 8 eksperymentów zakończone sukcesem!"
echo " Sprawdź folder /experiments/ aby zobaczyć pliki CSV."
echo "=================================================="