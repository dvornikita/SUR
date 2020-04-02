#!/usr/bin/env python3

import sys
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tabulate import tabulate

from data.meta_dataset_reader import (MetaDatasetEpisodeReader,
                                      TRAIN_METADATASET_NAMES,
                                      ALL_METADATASET_NAMES)
from models.model_utils import CheckPointer, sigmoid, cosine_sim
from models.model_helpers import get_domain_extractors
from models.losses import prototype_loss
from models.sur import apply_selection, sur
from models.models_dict import DATASET_MODELS_DICT
from utils import device
from config import args


def main():
    LIMITER = 5
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # Setting up datasets
    extractor_domains = TRAIN_METADATASET_NAMES
    all_test_datasets = ALL_METADATASET_NAMES
    loader = MetaDatasetEpisodeReader('test',
                                      train_set=extractor_domains,
                                      validation_set=extractor_domains,
                                      test_set=all_test_datasets)

    # define the embedding method
    dataset_models = DATASET_MODELS_DICT[args['model.backbone']]
    embed_many = get_domain_extractors(extractor_domains, dataset_models, args)

    accs_names = ['SUR']

    all_accs = dict()
    # Go over all test datasets
    for test_dataset in all_test_datasets:
        print(test_dataset)
        all_accs[test_dataset] = {name: [] for name in accs_names}

        with tf.compat.v1.Session(config=config) as session:
            for idx in tqdm(range(LIMITER)):
                # extract image features and labels
                sample = loader.get_test_task(session, test_dataset)
                context_features_dict = embed_many(sample['context_images'])
                target_features_dict = embed_many(sample['target_images'])
                context_labels = sample['context_labels'].to(device)
                target_labels = sample['target_labels'].to(device)

                # optimize selection parameters and perform feature selection
                selection_params = sur(context_features_dict, context_labels, max_iter=40)
                selected_context = apply_selection(context_features_dict, selection_params)
                selected_target = apply_selection(target_features_dict, selection_params)

                final_acc = prototype_loss(selected_context, context_labels,
                                           selected_target, target_labels)[1]['acc']
                all_accs[test_dataset]['SUR'].append(final_acc)

    # Make a nice accuracy table
    rows = []
    for dataset_name in all_test_datasets:
        row = [dataset_name]
        for model_name in accs_names:
            acc = np.array(all_accs[dataset_name][model_name]) * 100
            mean_acc = acc.mean()
            conf = (1.96 * acc.std()) / np.sqrt(len(acc))
            row.append(f"{mean_acc:0.2f} +- {conf:0.2f}")
        rows.append(row)

    table = tabulate(rows, headers=['model \\ data'] + accs_names, floatfmt=".2f")
    print(table)
    print("\n")


if __name__ == '__main__':
    main()
