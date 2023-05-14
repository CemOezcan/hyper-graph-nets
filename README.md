# HyperGraphNet
Implementation of different graph neural network architectures for estimating 
mesh-based pyhsics simulators.

## Results
![Alt Text](<img src="https://github.com/CemOezcan/hyper-graph-nets/blob/demo/demo/flag_base.gif" width="40" height="40" />)

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
