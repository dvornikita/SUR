#!/bin/bash

################### Training independent feature extractors ###################
./scripts/train_networks.sh


################################ Testing SUR ################################
python test.py --model.backbone=resnet18
