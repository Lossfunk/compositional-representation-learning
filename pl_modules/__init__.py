from .HyperNetworkSpatialEncoder import HyperNetworkSpatialEncoder
from .PESpatialEncoder import PESpatialEncoder

def get_module(config):
    model_type = config['model']['type']

    if model_type == 'PESpatialEncoder':
        return PESpatialEncoder(config)
    else:
        raise ValueError(f"Model type {model_type} not implemented.")