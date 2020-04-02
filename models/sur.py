import torch
import numpy as np

from models.model_utils import sigmoid, cosine_sim
from models.losses import prototype_loss
from utils import device


def apply_selection(features_dict, lambdas, normalize=True):
    """
    Performs masking of features via pointwise multiplying by lambda
    """
    lambdas_01 = sigmoid(lambdas)
    features_list = list(features_dict.values())
    if normalize:
        features_list = [f / (f ** 2).sum(-1, keepdim=True).sqrt()
                         for f in features_list]
    n_cont = features_list[0].shape[0]
    concat_feat = torch.stack(features_list, -1)
    return (concat_feat * lambdas_01).reshape([n_cont, -1])


def sur(context_features_dict, context_labels, max_iter=40):
    """
    SUR method: optimizes selection parameters lambda
    """
    lambdas = torch.zeros([1, 1, len(context_features_dict)]).to(device)
    lambdas.requires_grad_(True)
    n_classes = len(np.unique(context_labels.cpu().numpy()))
    optimizer = torch.optim.Adadelta([lambdas], lr=(3e+3 / n_classes))

    for i in range(max_iter):
        optimizer.zero_grad()
        selected_features = apply_selection(context_features_dict, lambdas)
        loss, stat, _ = prototype_loss(selected_features, context_labels,
                                       selected_features, context_labels)

        loss.backward()
        optimizer.step()
    return lambdas
