#!/bin/bash

################### Training independent feature extractors ###################
function train_fn {
    python train.py --model.name=$1 --data.train $2 --data.val $2 --data.test $2 --train.batch_size=$3 --train.learning_rate=$4 --train.max_iter=$5 --train.cosine_anneal_freq=$6 --train.eval_freq=$6
}

# Train base feature extractor on ImageNet
NAME="imagenet-net"; TRAINSET="ilsvrc_2012"; BATCH_SIZE=64; LR="3e-2"; MAX_ITER=480000; ANNEAL_FREQ=48000
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

#Then, train domain specific FiLM layers on every other dataset
# Omniglot
NAME="omniglot-film"; TRAINSET="omniglot"; BATCH_SIZE=16; LR="3e-2"; MAX_ITER=40000; ANNEAL_FREQ=3000
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

# Aircraft
NAME="aircraft-film"; TRAINSET="aircraft"; BATCH_SIZE=32; LR="1e-2"; MAX_ITER=30000; ANNEAL_FREQ=1500
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

# Birds
NAME="birds-film"; TRAINSET="cu_birds"; BATCH_SIZE=16; LR="3e-2"; MAX_ITER=40000; ANNEAL_FREQ=1500
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

# Textures
NAME="textures-film"; TRAINSET="dtd"; BATCH_SIZE=16; LR="3e-2"; MAX_ITER=40000; ANNEAL_FREQ=1500
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

# Quick Draw
NAME="quickdraw-film"; TRAINSET="quickdraw"; BATCH_SIZE=32; LR="1e-2"; MAX_ITER=400000; ANNEAL_FREQ=15000
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

# Fungi
NAME="fungi-film"; TRAINSET="fungi"; BATCH_SIZE=32; LR="1e-2"; MAX_ITER=400000; ANNEAL_FREQ=15000
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

# VGG Flower
NAME="vgg_flower-film"; TRAINSET="vgg_flower"; BATCH_SIZE=16; LR="1e-2"; MAX_ITER=30000; ANNEAL_FREQ=3000
train_fn $NAME $TRAINSET $BATCH_SIZE $LR $MAX_ITER $ANNEAL_FREQ

echo "All Feature Extractors are trained!"
