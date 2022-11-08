#!/bin/bash
EPOCHS=200
BATCH=250
DEVICE="cuda"
for seed in 1 2 3
do
    for model in "resnet_8_8"
        do
        python classification_image.py --model $model --method constrained --dataset translated_cifar10 --n_epochs $EPOCHS --device cuda:1 --n_samples_aug 2 --save --batch_size $BATCH --seed $seed --n_epochs_burnin 10
        python classification_image.py --model $model --method constrained --dataset cifar10_r180 --n_epochs $EPOCHS --device cuda:1 --n_samples_aug 2 --save --batch_size $BATCH --seed $seed --n_epochs_burnin 10
        python classification_image.py --model $model --method constrained --dataset cifar10_r90 --n_epochs $EPOCHS --device cuda:1 --n_samples_aug 2 --save --batch_size $BATCH --seed $seed --n_epochs_burnin 10
        python classification_image.py --model $model --method constrained --dataset scaled_cifar10  --n_epochs $EPOCHS --device cuda:1 --n_samples_aug 2 --save --batch_size $BATCH --seed $seed --n_epochs_burnin 10
        python classification_image.py --model $model --method constrained --dataset cifar10  --n_epochs $EPOCHS --device cuda:1 --n_samples_aug 2 --save --batch_size $BATCH --seed $seed --n_epochs_burnin 10
        for sz in 1000 5000 10000 20000
            do
            python classification_image.py --model $model --subset_size $sz --method constrained --dataset translated_cifar10  --n_epochs $EPOCHS --device cuda:1 --n_samples_aug 2 --save --batch_size $BATCH --seed $seed --n_epochs_burnin 10
            python classification_image.py --model $model --subset_size $sz --method constrained --dataset cifar10_r180 --n_epochs $EPOCHS --device cuda:1 --n_samples_aug 2 --save --batch_size $BATCH --seed $seed --n_epochs_burnin 10
            python classification_image.py --model $model --subset_size $sz --method constrained --dataset cifar10_r90 --n_epochs $EPOCHS --device cuda:1 --n_samples_aug 2 --save --batch_size $BATCH --seed $seed --n_epochs_burnin 10
            python classification_image.py --model $model --subset_size $sz --method constrained --dataset scaled_cifar10  --n_epochs $EPOCHS --device cuda:1 --n_samples_aug 2 --save --batch_size $BATCH --seed $seed --n_epochs_burnin 10
            python classification_image.py --model $model --subset_size $sz --method constrained --dataset cifar10  --n_epochs $EPOCHS --device cuda:1 --n_samples_aug 2 --save --batch_size $BATCH --seed $seed --n_epochs_burnin 10
            done
        done
    done
done