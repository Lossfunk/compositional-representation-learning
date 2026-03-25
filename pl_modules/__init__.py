from .HierarchicalBoxVAE import HierarchicalBoxVAE
from .SubspaceConceptLattice import SubspaceConceptLattice, MemorySubspaceConceptLattice


def get_module(config):
    model_type = config["model"]["type"]

    if model_type == "HierarchicalBoxVAE":
        return HierarchicalBoxVAE(config)
    elif model_type == "SubspaceConceptLattice":
        return SubspaceConceptLattice(config)
    elif model_type == "MemorySubspaceConceptLattice":
        return MemorySubspaceConceptLattice(config)
    else:
        raise ValueError(f"Model type {model_type} not implemented.")
