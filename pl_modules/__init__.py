from .HierarchicalBoxVAE import HierarchicalBoxVAE


def get_module(config):
    model_type = config["model"]["type"]

    if model_type == "HierarchicalBoxVAE":
        return HierarchicalBoxVAE(config)
    else:
        raise ValueError(f"Model type {model_type} not implemented.")
