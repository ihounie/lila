#!/bin/bash
N_SAMPLES=2
for SEED in 1 2 3
do
python classification_image.py --method augerino --dataset translated_fmnist --n_epochs 300 --device cuda --n_samples_aug $N_SAMPLES --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed $SEED --model cnn --project ALL_Inv_Augerino
python classification_image.py --method augerino --dataset fmnist_r180 --n_epochs 300 --device cuda --n_samples_aug $N_SAMPLES --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed $SEED --model cnn
python classification_image.py --method augerino --dataset fmnist_r90 --n_epochs 300 --device cuda --n_samples_aug $N_SAMPLES --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed $SEED --model cnn
python classification_image.py --method augerino --dataset scaled_fmnist --n_epochs 300 --device cuda --n_samples_aug $N_SAMPLES --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed $SEED --model cnn
python classification_image.py --method augerino --dataset fmnist --n_epochs 300 --device cuda --n_samples_aug $N_SAMPLES --save --optimize_aug --approx ggn_kron --batch_size 1000 --seed $SEED --model cnn
done