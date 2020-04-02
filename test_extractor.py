"""
This code allows you to evaluate performance of a single feature extractor + NCC
on several dataset.

For example, to test a resnet18 feature extractor trained on cu_birds
(that you downloaded) on test splits of ilsrvc_2012, dtd, vgg_flower, quickdraw, run:
python ./test_extractor.py --model.name=birds-net --model.backbone=resnet18 --data.test ilsrvc_2012 dtd vgg_flower quickdraw
"""

import os
import torch
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

from models.losses import prototype_loss
from models.model_utils import CheckPointer
from models.model_helpers import get_model
from data.meta_dataset_reader import (MetaDatasetEpisodeReader, MetaDatasetBatchReader)
from config import args


def main():
    TEST_SIZE = 600

    # Setting up datasets
    trainsets, valsets, testsets = args['data.train'], args['data.val'], args['data.test']
    test_loader = MetaDatasetEpisodeReader('test', trainsets, valsets, testsets)
    model = get_model(None, args)
    checkpointer = CheckPointer(args, model, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)
    model.eval()

    accs_names = ['NCC']
    var_accs = dict()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as session:
        # go over each test domain
        for dataset in testsets:
            print(dataset)
            var_accs[dataset] = {name: [] for name in accs_names}

            for i in tqdm(range(TEST_SIZE)):
                with torch.no_grad():
                    sample = test_loader.get_test_task(session, dataset)
                    context_features = model.embed(sample['context_images'])
                    target_features = model.embed(sample['target_images'])
                    context_labels = sample['context_labels']
                    target_labels = sample['target_labels']
                    _, stats_dict, _ = prototype_loss(
                        context_features, context_labels,
                        target_features, target_labels)
                var_accs[dataset]['NCC'].append(stats_dict['acc'])

    # Print nice results table
    rows = []
    for dataset_name in testsets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(var_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")


if __name__ == '__main__':
    main()



