# MemorySubspaceLattice

## Overview
- In this line of experimentation, we are developing an algorithm which:
    - In unsupervised/self-supervised in nature
    - Learns representations which are *compositional* and *hierarchical* in nature
    - Is heavily based on Formal Concept Analysis (FCA), we essentially need the learned representation space to be a *concept lattice*.

## Algorithm Details
- There are broadly three main processing components:
    - A perceptual encoder which maps images to a latent space.
    - A concept encoder which maintains a conceptual latent space (comprising of attribute and object subspaces).
    - A decoder which reconstructs the original images from the latent space/concept space (usually only the attribute subspace)
- The key idea is that concepts exist in a lattice structure, each concept is defined by a set of attributes and a set of objects. 
    - We are representing these concepts as tuples of (attribute_subspace, object_subspace).
    - Each subspace is defined as a set of basis vectors, which span that subspace.
    - In a FCA lattice, the more the attributes, the smaller the object space (intersection), and the more the objects, the smaller the attribute space (intersection). This is defined as a galois connection between the two spaces
- Here, we are trying to learn these subspaces from scratch using a neural network. A key idea we are building upon is that combining concepts will lead to the *lowest common concept*, which is the intersection of the attribute subspaces and union of the object subspaces.
    - As we combine more concepts, we generally end up with smaller attribute subspaces (because there are lesser common attributes between them) and larger object subspaces (because there are more objects being considered).
    - With respect to the computed subspaces, we are measuring this notion of size as the *rank* of the subspace (which is the number of unique basis vectors required to span the subspace).
- Overall, the algorithm is broadly as follows:
    - We sample a batch of images, and pass them through the perceptual encoder to get the perceptual latent space.
    - We create combinations, ranging from cardinality 1 to N (where N is the number of images in the batch).
    - For each combination, we compute the *attribute subspace* and *object subspace*.
    - We take the attribute subspace for the singleton concepts (single image concepts) and try to reconstruct the image from it using a decoder.
    - Objective functions:
        - Reconstruction loss
        - Galois attribute and object loss (combined concept's attribute space is included in each of the individual concepts' attribute spaces, and each individual concept's object space is included in the combined concept's object space)
        - Intersection and union consistency loss (intersection of attribute spaces and union of object spaces)
        - Singleton structural loss (singleton attribute concepts have rank of 2, singleton object concepts have rank of 1)
        - Other regularization losses that we are still exploring
- In MemorySubspaceLattice, we are using a *memory* module which is responsible for maintaining the attribute and object subspaces. It is similar to a VQVAE, where we have a memory bank of attribute and object subspaces. We use commitment and utilization losses to ensure the concept encoder's output is close to the memory bank, and that the memory bank is utilized well.
- We apply some structural losses to the memory bank itself:
    - Slot and obj orthogonality. The entries in each slot are orthogonal to each other. All the object subspaces are orthogonal to each other.
    - Rank constraints.

## Required Conditions
- The dataset comprises of 4 different images: 
    - Red Circle
    - Red Square
    - Blue Circle
    - Blue Square
- Script for generating the dataset: /mnt/home/ubuntu/workspace/code/compositional-representation-learning/datasets/v0_dataset.py
- The algorithm should ingest this data and learn representations which are *compositional* and *hierarchical* in nature.
- The exact conditions we will use to measure this are as follows:
    - A perfect lattice structure which has the correct inclusions close to 1 and anti-inclusions close to 0.
    - Appropriate rank ranges in the attribute space (0-2) and object space (1-4).
        - Singleton attribute concepts have rank of 2, since they are a combination of 2 concepts (e.g. red + circle = red circle).
        - Singleton object concepts have rank of 1, since they are the base concepts.
        - When we combine two concepts which share *1* attribute:
            - The attribute rank should be 1 (the combination's attribute space will be an intersection of the two concepts' attribute spaces, which will have rank 1).
            - The object rank should be 2 (the combination's object space will be the union of the two concepts' object spaces, which will have rank 2).
        - When we combine two concepts which share *0* attributes:
            - The attribute rank should be 0 (the combination's attribute space will be an intersection of the two concepts' attribute spaces, which will have rank 0).
            - The object rank should be 4 (the combination's object space will be the union of the two concepts' object spaces, which will have rank 4). For now, even rank 2/3 is acceptable since we are not enforcing maximal concepts.
    - Perfect reconstruction of the original images from the learned representations.
            

    