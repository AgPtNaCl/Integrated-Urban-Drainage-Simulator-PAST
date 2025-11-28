# Process-aware Spatio-Temporal Graph-based Surrogate

This repository provides the official implementation of our study **"Exploring a Process-aware Spatiotemporal Graph-based Surrogate for Integrated Urban Drainage Simulation"**.

The proposed PAST (Process-Aware Spatiotemporal model) is a graph-based surrogate that enables real-time simulation of urban drainage systems. It captures the complete rainfall–runoff–routing process while considering boundary inflows and rule-based control strategies. By integrating process-awareness into spatiotemporal graph learning, PAST offers a fast and accurate alternative to conventional hydrodynamic models, making it suitable for both research and practical applications in urban water management.

For the original implementation of Graph attention network (GAT), please refer to the paper Graph Attention Networks, ICLR 2018. (https://doi.org/10.48550/arXiv.1710.10903)

# Requirements

python 3.8.18 and packages in requirements.txt

# How to use
training
```
python train.py
```
The file that records the model structure is stored in the ```model``` folder.  Details of the model and training parameters can refer to ```config.yaml```. The default model is PAST. The target training model can be changed by modifying the model name parameter in the command line.

