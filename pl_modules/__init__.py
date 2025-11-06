from .HyperNetworkSpatialEncoder import HyperNetworkSpatialEncoder
from .PESpatialEncoder import PESpatialEncoder
from .PatchBoxEmbeddings import PatchBoxEmbeddings
from .PatchBoxEmbeddingsVAE import PatchBoxEmbeddingsVAE

def get_module(config):
    model_type = config['model']['type']

    if model_type == 'PESpatialEncoder':
        return PESpatialEncoder(config)
    elif model_type == 'PatchBoxEmbeddings':
        return PatchBoxEmbeddings(config)
    elif model_type == 'PatchBoxEmbeddingsVAE':
        return PatchBoxEmbeddingsVAE(config)
    else:
        raise ValueError(f"Model type {model_type} not implemented.")