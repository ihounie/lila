#!/bin/bash
for seed in 1 1 3
do
    for model in "mlp" "cnn"
        do
        python classification_image.py --model $model --method uniform_aug --dataset translated_mnist --n_epochs 1000 --device cuda --n_samples_aug 1 --save --batch_size 1000 --seed $seed
        python classification_image.py --model $model --method uniform_aug --dataset mnist_r180 --n_epochs 1000 --device cuda --n_samples_aug 1 --save --batch_size 1000 --seed $seed
        python classification_image.py --model $model --method uniform_aug --dataset mnist_r90 --n_epochs 1000 --device cuda --n_samples_aug 1 --save --batch_size 1000 --seed $seed
        python classification_image.py --model $model --method uniform_aug --dataset scaled_mnist --n_epochs 1000 --device cuda --n_samples_aug 1 --save --batch_size 1000 --seed $seed
        python classification_image.py --model $model --method uniform_aug --dataset mnist --n_epochs 1000 --device cuda --n_samples_aug 1 --save --batch_size 1000 --seed $seed
        for sz in 311 1250 5000 20000
            do
            python classification_image.py --model $model --subset_size $sz --method uniform_aug --dataset translated_mnist --n_epochs 1000 --device cuda --n_samples_aug 1 --save --batch_size 1000 --seed $seed
            python classification_image.py --model $model --subset_size $sz --method uniform_aug --dataset mnist_r180 --n_epochs 1000 --device cuda --n_samples_aug 1 --save --batch_size 1000 --seed $seed
            python classification_image.py --model $model --subset_size $sz --method uniform_aug --dataset mnist_r90 --n_epochs 1000 --device cuda --n_samples_aug 1 --save --batch_size 1000 --seed $seed
            python classification_image.py --model $model --subset_size $sz --method uniform_aug --dataset scaled_mnist --n_epochs 1000 --device cuda --n_samples_aug 1 --save --batch_size 1000 --seed $seed
            python classification_image.py --model $model --subset_size $sz --method uniform_aug --dataset mnist --n_epochs 1000 --device cuda --n_samples_aug 1 --save --batch_size 1000 --seed $seed
            done
        done
    done
done