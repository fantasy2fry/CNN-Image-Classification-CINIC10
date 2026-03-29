#!/bin/bash

# Zatrzymuje skrypt, jeśli któryś z eksperymentów wyrzuci błąd
set -e

# Zmienne ogólne
EPOCHS=30
SCRIPT="python -m src.train_proto"

echo "============================================================"
echo " Uruchamiam serię eksperymentów dla Prototypical Networks!  "
echo " Liczba epok dla każdego eksperymentu: $EPOCHS"
echo "============================================================"

# --- 1. BAZA: Wpływ liczby próbek (Few-Shot vs More-Shot) ---
# Kluczowe pytanie dla ProtoNet: ile próbek per klasa wystarczy?

echo -e "\n---> [1/9] Baza: 5-shot (minimum)"
$SCRIPT --epochs $EPOCHS --samples_per_class 5

echo -e "\n---> [2/9] Baza: 10-shot (default)"
$SCRIPT --epochs $EPOCHS --samples_per_class 10

echo -e "\n---> [3/9] Baza: 50-shot (więcej danych)"
$SCRIPT --epochs $EPOCHS --samples_per_class 50


# --- 2. FREEZE vs FINE-TUNE ---
# Przy małych zbiorach zamrożenie warstw bywa lepsze niż pełny fine-tuning

echo -e "\n---> [4/9] Frozen backbone (tylko layer4 + głowica trenowana)"
$SCRIPT --epochs $EPOCHS --samples_per_class 10 --freeze_features

echo -e "\n---> [5/9] Frozen + więcej danych (50-shot)"
$SCRIPT --epochs $EPOCHS --samples_per_class 50 --freeze_features


# --- 3. PRZESTRZEŃ EMBEDDINGÓW ---
# Mniejszy embedding = bardziej kompaktowy, większy = więcej informacji

echo -e "\n---> [6/9] Mała przestrzeń embeddingów (64-dim)"
$SCRIPT --epochs $EPOCHS --samples_per_class 10 --embedding_dim 64

echo -e "\n---> [7/9] Duża przestrzeń embeddingów (256-dim)"
$SCRIPT --epochs $EPOCHS --samples_per_class 10 --embedding_dim 256


# --- 4. REGULARYZACJA: Walka z overfittingiem na małym zbiorze ---

echo -e "\n---> [8/9] Cutout + wyższy dropout (mocna augmentacja)"
$SCRIPT --epochs $EPOCHS --samples_per_class 10 --use_cutout --dropout 0.5


# --- 5. ULTIMATE: Najlepsze ustawienia ze wszystkich sekcji ---

echo -e "\n---> [9/9] ULTIMATE: 50-shot + Frozen + Cutout + Weight Decay"
$SCRIPT --epochs $EPOCHS --samples_per_class 50 --freeze_features --use_cutout --weight_decay 1e-4 --embedding_dim 128 --dropout 0.3

echo -e "\n============================================================"
echo " Wszystkie 9 eksperymentów zakończone sukcesem!"
echo " Sprawdź folder /experiments/ aby zobaczyć pliki CSV."
echo "============================================================"