from .HyperNetworkSpatialEncoder import HyperNetworkSpatialEncoder
from .PESpatialEncoder import PESpatialEncoder
from .BoxEmbeddings import PatchBoxEmbeddingsVAE, PatchBoxEmbeddings, HierarchicalBoxEmbeddingsVAE
from .HierarchicalBoxVAE import HierarchicalBoxVAE


def get_module(config):
    model_type = config["model"]["type"]

    if model_type == "PESpatialEncoder":
        return PESpatialEncoder(config)
    elif model_type == "PatchBoxEmbeddings":
        return PatchBoxEmbeddings(config)
    elif model_type == "PatchBoxEmbeddingsVAE":
        return PatchBoxEmbeddingsVAE(config)
    elif model_type == "HierarchicalBoxEmbeddingsVAE":
        return HierarchicalBoxEmbeddingsVAE(config)
    elif model_type == "HierarchicalBoxVAE":
        return HierarchicalBoxVAE(config)
    else:
        raise ValueError(f"Model type {model_type} not implemented.")
