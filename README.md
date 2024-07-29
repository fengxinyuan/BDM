# Bootstrap Deep Metric for Seed Expansion in Attributed Networks
This repository is an implementation of a graph deep metric learning framework for seed expansion problems:
## Overview of BDM framework:
![image](https://github.com/user-attachments/assets/52d234a0-84b1-4631-9866-445967f606bc)
*This figure is overview of BDM framework. Upper layer: the main idea to use deep metric for seed expansion. We map the network’s topological structure and node content into a representation space such that community members are close to each other while non-members are distant. Subsequently, we select the nodes near the approximated community center as the expansion nodes. Lower layer: BDM trains its main mapping 𝑓𝜃 by minimizing metric losses between the produced positive-anchor 𝑎𝑃 and representations of positive nodes in 𝐻e𝜃, as well as between self-anchors 𝐻e𝜙 and representations of each node in 𝐻e𝜃, where 𝜃 and 𝜙 are learnable parameters. The parameter 𝜙 of auxiliary mapping 𝑓𝜙 is updated as a slow-moving average of 𝜃. Here, sg means stop-gradient.*

## Requirements
- numpy==1.24.3
- scikit-learn==1.3.1
- torch==2.0.1
- torch_geometric==2.3.0


## Usage
run BDM demo by: 
```python run_bdm.py -d 'Cora' -c 0 -s 1 -u 'False'``` 
where -d represents the used dataset, and -c denotes the index of label to be used as positive, -s indicates the size of the random seed, and -u indicates whether to use a high-dimensional node as the seed node.

## Change Log
To be updated
## Reference
To be updated
