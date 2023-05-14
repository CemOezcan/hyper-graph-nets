# HyperGraphNet
Learned physics simulators utilizing Graph Neural Networks (GNNs) achieve faster inference times and have enhanced generalization capabilities when compared to classical physics simulators. However, the ability of GNNs to capture long-range dependencies is limited by a constant number of GNN layers (or message passing layers). To overcome this issue, we designed HyperGraphNets, a framework for GNNs that uses remote message passing to facilitate modeling long-range dependencies.

This repository contains the implementation of multiple GNN architectures as learned physics simulators.

## Examples
The following simulations and error curves illustrate the enhanced remote message passing capabilities of HyperGraphNets (left) in comparison to the baseline, MeshGraphNets (right). MP denotes the number of message passing layers.

### Deforming Plate
<p float="middle">
<img src="https://github.com/CemOezcan/hyper-graph-nets/blob/demo/demo/plate_hgn.gif" width="400" height="225" />
<img src="https://github.com/CemOezcan/hyper-graph-nets/blob/demo/demo/pate_base.gif" width="400" height="225" />
</p>

<p float="middle">
<img src="https://github.com/CemOezcan/hyper-graph-nets/blob/demo/demo/plate_10.png" width="400" height="225" />
<img src="https://github.com/CemOezcan/hyper-graph-nets/blob/demo/demo/plate_rollout.png" width="400" height="225" />
</p>


### Flag Simple
<p float="middle">
<img src="https://github.com/CemOezcan/hyper-graph-nets/blob/demo/demo/flag_spectral.gif" width="400" height="225" />
<img src="https://github.com/CemOezcan/hyper-graph-nets/blob/demo/demo/flag_base.gif" width="400" height="225" />
</p>
<img src="https://github.com/CemOezcan/hyper-graph-nets/blob/demo/demo/flag_rollout.png" width="400" height="225" />


## Setting up the environment
This project uses [PyPI](https://pypi.org/) for handling packages
and dependencies. To get started, we recommend creating a new virtual environment and installing the required 
dependencies using `pip install -r \Path\to\requirements.txt`. 

##  Recording
We provide logging of metrics and visualizations to [W&B](https://wandb.ai). 
This requires logging in to [W&B](https://wandb.ai) via `wandb login` 
(For more information, read the [quickstart guide of W&B](https://docs.wandb.ai/quickstart)).

## Downloading datasets
Download a dataset to its respective sub folder within the data directory `./data`. 
Let's consider the `flag_minimal` dataset: 
* Download the data set via `bash download.sh flag_simple data/flag_minimal`
* Move the contents of the downloaded `data/flag_minimal/flag_minimal` directory into `data/flag_minimal/input`
* In `data/flag_minimal/input`, execute `python -m tfrecord.tools.tfrecord2idx <file>.tfrecord <file>.idx` 
for `train.tfrecord`, `test.tfrecord` and `valid.tfrecord` respectively.

## Creating an experiment
The folder `configs` contains a number of `.yaml` files that describe the configuration of the task to run. 
To run an experiment from any of these files on a local machine, type
`python main.py "${CONFIG}"`, where `${CONFIG}` refers to the name of the config file (without the suffix `.yaml`).

## Folder structure
    .
    ├── config                    # Config files for setting up experiments
    ├── data                      # Data sets, models and plots
        ├── cylinder_flow         # Input and output data for the cylinder_flow task
            ├── input 
            ├── output 
        ├── deforming_plate       # Input and output data for the deforming_plate task
            ├── input 
            ├── output 
        ├── flag_minimal          # Input and output data for the flag_minimal task
            ├── input 
            ├── output 
        ├── flag_simple           # Input and output data for the flag_simple task
            ├── input 
            ├── output
    ├── src                       # Source code
        ├── algorithms            # Training and evaluation of GNN-based physics simulators
        ├── data                  # Loading and preprocessing raw data sets
        ├── graph_balancer        # Graph balancing algorithms for remote message passing
        ├── migration             # PyTorch implementation of MeshGraphNets and its extension, HyoerGraphNets
        ├── model                 # Task specific Graph Neural Networks
        ├── rmp                   # Graph clustering algorithms for remote message passing
        ├── tasks                 # Wrapper methods for training, evaluating and visualizing 
    ├── wandb                     # Recording via W&B
    ├── .gitignore                 
    ├── requirements.txt           
    ├── main.py
    ├── download.sh 
    ├── LICENSE
    └── README.md
