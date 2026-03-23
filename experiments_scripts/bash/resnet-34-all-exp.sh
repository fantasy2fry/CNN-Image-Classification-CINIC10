#!/bin/bash

# Zatrzymuje skrypt, jeśli któryś z modeli wyrzuci błąd
set -e

# Zmienne ogólne
EPOCHS=30
SCRIPT="python -m src.train"

echo "======================================================"
echo " Uruchamiam serię eksperymentów dla ResNet-34! "
echo " Cel: Transfer Learning [cite: 34] vs Pełny Fine-Tuning "
echo " Liczba epok dla każdego modelu: $EPOCHS"
echo "======================================================"

# --- 1. TRANSFER LEARNING (ZAMROŻONE WAGI): Tylko ostatnia warstwa  ---
# Tutaj możemy użyć standardowego LR (0.001), bo uczymy sieć od zera, ale tylko klasyfikator
echo -e "\n---> [1/8] Transfer Learning: Adam (LR=0.001)"
$SCRIPT --model resnet34 --freeze_features --optimizer adam --lr 0.001 --epochs $EPOCHS

echo -e "\n---> [2/8] Transfer Learning: SGD (LR=0.01)"
$SCRIPT --model resnet34 --freeze_features --optimizer sgd --lr 0.01 --epochs $EPOCHS


# --- 2. PEŁNY FINE-TUNING (ODMROŻONE WAGI): Cała sieć się uczy  ---
# Bardzo ważne: Używamy małego LR (np. 0.0001), aby delikatnie dostosować mądre wagi z ImageNetu!
echo -e "\n---> [3/8] Fine-Tuning: Adam (LR=0.0001)"
$SCRIPT --model resnet34 --optimizer adam --lr 0.0001 --epochs $EPOCHS

echo -e "\n---> [4/8] Fine-Tuning: SGD (LR=0.001 - zmniejszone w stosunku do bazy)"
$SCRIPT --model resnet34 --optimizer sgd --lr 0.001 --epochs $EPOCHS


# --- 3. REGULARYZACJA W TRANSFER LEARNINGU ---
# Sprawdzamy, czy Weight Decay i Cutout pomogą przy zamrożonej sieci
echo -e "\n---> [5/8] Transfer Learning + Weight Decay (Adam, L2=1e-4)"
$SCRIPT --model resnet34 --freeze_features --optimizer adam --lr 0.001 --weight_decay 0.0001 --epochs $EPOCHS

echo -e "\n---> [6/8] Transfer Learning + Cutout (Adam)"
$SCRIPT --model resnet34 --freeze_features --optimizer adam --lr 0.001 --use_cutout --epochs $EPOCHS


# --- 4. ZAAWANSOWANY FINE-TUNING (KOMBO) ---
echo -e "\n---> [7/8] Fine-Tuning + Weight Decay + Cutout (Adam, LR=0.0001)"
$SCRIPT --model resnet34 --optimizer adam --lr 0.0001 --weight_decay 0.0001 --use_cutout --epochs $EPOCHS

echo -e "\n---> [8/8] Fine-Tuning + Weight Decay + Cutout (SGD, LR=0.001)"
$SCRIPT --model resnet34 --optimizer sgd --lr 0.001 --weight_decay 0.0001 --use_cutout --epochs $EPOCHS

echo -e "\n======================================================"
echo " Wszystkie 8 eksperymentów dla ResNet-34 zakończone sukcesem!"
echo " Sprawdź folder /experiments/ aby zobaczyć pliki CSV."
echo "======================================================"