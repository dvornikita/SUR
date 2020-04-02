import os
import gin
import torch
from functools import partial

from models.model_utils import CheckPointer
from models.models_dict import DATASET_MODELS_RESNET18
from utils import device
from paths import PROJECT_ROOT


def get_model(num_classes, args):
    train_classifier = args['model.classifier']
    model_name = args['model.backbone']
    dropout_rate = args.get('model.dropout', 0)

    if 'pnf' in model_name:
        from models.resnet18_pnf import resnet18

        base_network_name = DATASET_MODELS_RESNET18['ilsvrc_2012']
        base_network_path = os.path.join(PROJECT_ROOT, 'weights', base_network_name, 'model_best.pth.tar')
        model_fn = partial(resnet18, dropout=dropout_rate,
                           pretrained_model_path=base_network_path)
    else:
        from models.resnet18 import resnet18
        model_fn = partial(resnet18, dropout=dropout_rate)

    model = model_fn(classifier=train_classifier,
                     num_classes=num_classes,
                     global_pool=False)
    model.to(device)
    return model


def get_optimizer(model, args, params=None):
    learning_rate = args['train.learning_rate']
    weight_decay = args['train.weight_decay']
    optimizer = args['train.optimizer']
    params = model.parameters() if params is None else params
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
    elif optimizer == 'momentum':
        optimizer = torch.optim.SGD(params,
                                    lr=learning_rate,
                                    momentum=0.9, nesterov=args['train.nesterov_momentum'],
                                    weight_decay=weight_decay)
    else:
        assert False, 'No such optimizer'
    return optimizer


def get_domain_extractors(trainset, dataset_models, args):
    if 'pnf' in args['model.backbone']:
        return get_pnf_extractor(trainset, dataset_models, args)
    else:
        return get_multinet_extractor(trainset, dataset_models, args)


def get_multinet_extractor(trainsets, dataset_models, args):
    extractors = dict()
    for dataset_name in trainsets:
        if dataset_name not in dataset_models:
            continue
        args['model.name'] = dataset_models[dataset_name]
        extractor = get_model(None, args)
        checkpointer = CheckPointer(args, extractor, optimizer=None)
        extractor.eval()
        checkpointer.restore_model(ckpt='best', strict=False)
        extractors[dataset_name] = extractor

    def embed_many(images, return_type='dict'):
        with torch.no_grad():
            all_features = dict()
            for name, extractor in extractors.items():
                all_features[name] = extractor.embed(images)
        if return_type == 'list':
            return list(all_features.values())
        else:
            return all_features
    return embed_many


def get_pnf_extractor(trainsets, dataset_models, args):
    film_layers = dict()
    for dataset_name in trainsets:
        if dataset_name not in dataset_models or 'ilsvrc' in dataset_name:
            continue
        ckpt_path = os.path.join(PROJECT_ROOT, 'weights', dataset_models[dataset_name],
                                 'model_best.pth.tar')
        state_dict = torch.load(ckpt_path, map_location=device)['state_dict']
        film_layers[dataset_name] = {k: v for k, v in state_dict.items()
                                     if 'cls' not in k}
        print('Loaded FiLM layers from {}'.format(ckpt_path))

    # define the base extractor
    base_extractor = get_model(None, args)
    base_extractor.eval()
    base_layers = {k: v for k, v in base_extractor.get_state_dict().items() if 'cls' not in k}

    # initialize film layers of base extractor to identity
    film_layers['ilsvrc_2012'] = {k: v.clone() for k, v in base_layers.items()}

    def embed_many(images, return_type='dict'):
        with torch.no_grad():
            all_features = dict()

            for domain_name in trainsets:
                # setting up domain-specific film layers
                domain_layers = film_layers[domain_name]
                for layer_name in base_layers.keys():
                    base_layers[layer_name].data.copy_(domain_layers[layer_name].data)

                # inference
                all_features[domain_name] = base_extractor.embed(images)
        if return_type == 'list':
            return list(all_features.values())
        else:
            return all_features
    return embed_many
