#!/bin/bash
LRDUAL=0.0005
EPOCHS=400
DEVICE="cuda:1"
for seed in 1 2 3
do  
    EPS=0.8
    for model in "mlp"
        do
        python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --method constrained --dataset translated_fmnist --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
        python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --method constrained --dataset fmnist_r180 --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
        python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --method constrained --dataset fmnist_r90 --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
        python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --method constrained --dataset scaled_fmnist --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
        done
    done
    EPS=0.6
    for model in "cnn"
        do
        python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --method constrained --dataset translated_fmnist --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
        python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --method constrained --dataset fmnist_r180 --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
        python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --method constrained --dataset fmnist_r90 --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
        python classification_image.py --lr_dual $LRDUAL --epsilon $EPS --model $model --method constrained --dataset scaled_fmnist --n_epochs $EPOCHS --device $DEVICE --n_samples_aug 2 --save --batch_size 1000 --seed $seed
        done
    done
done