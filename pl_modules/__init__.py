from .HyperNetworkSpatialEncoder import HyperNetworkSpatialEncoder
from .PESpatialEncoder import PESpatialEncoder
from .PatchBoxEmbeddings import PatchBoxEmbeddings

def get_module(config):
    model_type = config['model']['type']

    if model_type == 'PESpatialEncoder':
        return PESpatialEncoder(config)
    elif model_type == 'PatchBoxEmbeddings':
        return PatchBoxEmbeddings(config)
    else:
        raise ValueError(f"Model type {model_type} not implemented.")