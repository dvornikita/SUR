#!/bin/bash

mode='test'
backbone='resnet18_pnf'
for dataset in "ilsvrc_2012" "omniglot" "aircraft" "cu_birds" "dtd" "quickdraw" "fungi" "vgg_flower" " traffic_sign" "mscoco" "mnist" "cifar10" "cifar100"; do
    python ./data/create_features_db.py --data.train ilsvrc_2012 omniglot aircraft cu_birds dtd quickdraw fungi vgg_flower --data.val ${dataset} --data.test ${dataset} --model.backbone=${backbone} --dump.mode=${mode} --dump.size=600
done

