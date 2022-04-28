# Why should I care?
Developing machine learning algorithms is complicated. Without proper countermeasures,
projects quickly become messy over their lifespan, eventually leading to slower development and more accidental bugs. 
Even if things work out fine code-wise, running the "right" experiments and interpreting the results of a 
sufficiently complex method is often more art than science. 

This repository is a starting point for your project at the ALR. It natively integrates `wandb`, `cw2` and `optuna`, all
of which are packages that most of us use regularly, and that otherwise require some work to properly set it. We also
provide a flexible but extensive recording utility to allow for a range of different things to be tracked for each of
your runs.

Lastly, there are some testing templates provided to allow you to write quick and efficient unit tests.

# Getting Started
## Copying the template
Navigate to the main page of [this repository](https:github.com/ALRhub/ALRProject), 
select "Use this template" in the menu bar and follow the instructions afterwards. For more
information on how to create a repository from a template, see
[here](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/creating-a-repository-from-a-template)

## Setting up the environment
This project uses poetry (https://python-poetry.org/) and conda (https://docs.conda.io/en/latest/) for handling packages
and dependencies.
To get started, we recommend creating a new conda environment
`conda create -f environment.yaml` and then letting poetry do its
magic via `poetry init`. If you do not already have poetry installed, you can do so via 
`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`.

Currently, you also need to install cw2 and optunawork from our github repositories via
`pip install git+https://www.github.com/ALRhub/cw2.git` and
`pip install git+https://www.github.com/ALRhub/optunawork.git`

## Creating an experiment
Experiments are configured and distributed via cw2 (https://www.github.com/ALRhub/cw2.git). For this, the folder `configs` contains
a number of `.yaml` files that describe the configuration of the task to run. 
To run an experiment from any of these files on a local machine, type
`python main.py configs/$FILE_NAME$.yaml -e $EXPERIMENT_NAME$ -o`. To start an experiment on a cluster that uses Slurm
(https://slurm.schedmd.com/documentation.html), run 
`python main.py configs/$FILE_NAME$..yaml -e $EXPERIMENT_NAME$ -o -s --nocodecopy`.

Running an experiment essentially provides a (potentially nested) config dictionary that can be used to specify e.g., an
algorithm, a task and a dataset, each of which may or may not include their own parameters. For more information on how
to use this, refer to the cw2 docs.

## Managing packages
As hinted at in the "Getting started" section, we choose conda for our virtual environments. This helps to keep different
projects separate, and prevents accidental version mismatches or package conflicts.
For managing the packages inside the environment, we use poetry. In its most basic form, poetry lists the
requirements for a project (i.e., which external packages it uses) while making sure that all packages are compatible.
The files that poetry maintains are human-readable, and can also be used to quickly initialize a new environment.
New packages can be added using the `poetry add package_name` command, while `poetry remove package_name` removes an
existing package. For more information, please refer to https://python-poetry.org/.

## Extending the template
This project contains a template and a small but expressive working example 
(a `two_moons` classification task) of how to use it. 
It is meant as a starting point for your project.
To easily extend it, we provide a set of interfaces to work with.

* `src/tasks` contains the tasks that your approach is meant to tackle. The `AbstractTask` class defines some interfaces
  for new tasks that ensure that it is compatible with the main framework and the recording. These are
    * `run_iteration()` which is repeatedly called by the main and expects the task to perform a single iteration of your algorithm.
    * `get_scalars()` which provides scalar values such as losses, rewards and metrics for the task and algorithm at the
      current iteration. These scalars are maintained and plottes by the `ScalarsLogger`, and can also be
      used for Optuna optimization.
    * `plot()` that is called by the `VisualizationLogger` to plot a visualization of your task 
      (and the algorithm interacting with it) using a plotly figure.
* `src/algorithms` contains the algorithms/approaches to solve the given tasks. At runtime, the algorithm is *given to*
  your task. To keep an algorithm compatible with its task, it should implement
  * an `initialize(task_information)` function that can be called by the task before starting the first iteration to
    provide pre-processed information about the task. This information can e.g., be the task dimensionality, the
    kind of environment to deal with etc.
  * a `fit_iteration()`-routine that trains the algorithm for a single iteration. This may be a training epoch, a number
  of gradient updates or more generally anything that is repeatedly executed to optimize whatever the goal of the
    algorithm is.
  * a `predict()`-routine to evaluate given input data. This can e.g., be a forward pass of a network.
  * a `score()`-routine that outputs a dictionary of computed metrics and entities based on given input data. This
    can for example be your network loss and accuracy, statistics of your policy for a given environment, etc.
* `recording/loggers` has an `AbstractLogger` class that defines a basic logging routine. Every logger comes with
  * a writer that logs to console and a shared `out.txt` file.
  * a `log_iteration()`method that records whatever the logger is meant to record at the current iteration. This method
    may return a dictionary of computed values, which is then given to subsequent loggers as an input. This prevents
    repeated calculation of expensive metrics such as an expected reward or a numerical integration.
  * a `finalize()` method to finish up the recording at the end of the run. This can be used to write information to
    disc or make a final plot.
  * Additionally, every new logger needs to be registered in the `recording/register_loggers.py` file. Here, you can
    decide under which circumstances you want to use your new logger, allowing you to make it 
    e.g., task- or algorithm-dependent.
    
  
* The `configs` directory specifies a hierarchy of experiment configurations. 
  We recommend adding as many hyperparameters as possible to it to allow for easy hyperparameter search and tuning.
  * Your data_loader, task, algorithm and loggers all have access to the (potentially adapted) `params` subdict
    specified in the config for the current experiment via the `config` dictionary they receive as an input.
    * Similarly, you may want to run multiple iterations at a time without recording between each of them. For this,
      you can create another loop inside your tasks `run_iteration()`-method that is then executed for each iteration.


#### Caveats
* There is nothing magical about these interfaces. They are meant to be as flexible as possible while confining with 
  good coding practices, `cw2` and `optuna`. If you find that they do not work for your particular use-case, feel free
  to adapt them to your needs.
* This template is structured around the idea of an iterative algorithm, which is pretty much everything that has
  an outer loop. If your algorithm does not fit into this, you can work around it by setting the number of iterations
  to `1` and then doing the whole training routine in this iteration.

## Project Structure

###  Recording
We provide a logger for all console outputs, different scalar metrics and task-dependent visualizations per iteration. 
The scalar metrics may optionally be logged to the [wandb dashboard](https://wandb.ai).
The metrics and plot that are recorded depend on both the task and the algorithm being run. 
The loggers can (and should) be extended
to suite your individual needs, e.g., by plotting additional metrics or animations whenever necessary.

All locally recorded entities are saved the `reports` directory, where they are organized by their experiment and repetition.


### Configs
* `config` contains a hierarchy of configuration files for the experiments that are compatible with cw2. 
The `default.yaml` file is the most general, with other files building off of it to create more specific sets of
experiments. 
* An experiment configuration ("config") contains both information about the task at hand as well as the algorithm to run.
For more about how to configure config files, see the [cw2 logs](https://www.github.com/ALRhub/cw2).
* We additionally extend cw2 by a parameter `optuna` for general optuna configuration, and a parameter `optuna_hps` that handles the
parameters to optimize. If *both* are specified, the run is automatically wrapped in an optuna trial. This is
based on [optunawork](https://www.github.com/ALRhub/optunawork).

### Reports
`reports` contains everything that your loggers pick up. 
That may be simple logs of metrics like the loss and accuracy of your network, or more sophisticated things such as
visualizations of your tasks, plots over time and more.

### Source
The `source` folder contains the source code of this project. It is separated into `algorithms` and `tasks`, where
tasks may contain something like Gym Environments, that are optimized based on a given algorithm.

### Data
`data` is where datasets for the different experiments are stored. This directory holds the data itself, as well as a
`data_loader.py` file that takes a config (as a nested python dictionary) and returns the raw data specified
by the config.

### Tests
We provide basic unit testing utilities in `test.py`. The file will create a testSuite for all (potentially nested)
files in the `tests` directory. 
For a subdirectory to be registered for this, it must have an `__init__.py` file. For a class to be registered, it
must be a derivative of `unittest.TestCase`

#### Blacklists and whitelists
You can choose which tests to run by either blacklisting or whitelisting some of them. To this end,
the argument `-b arg1 arg2` removes all tests whose name, class, or (recursive) containing folder is in `[arg1, arg2]`.
Similarly, `-w` receives a list of tests to be whitelisted and allows only tests have on of the listed names, classes,
or (recursive) containing folders. 


