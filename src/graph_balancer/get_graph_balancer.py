"""
Utility class to select a graph balancer strategy based on a given config file
"""
from src.graph_balancer.abstract_graph_balancer import AbstractGraphBalancer
from src.graph_balancer.graph_balancer import GraphBalancer
from src.graph_balancer.random_balancing import RandomGraphBalancer
from src.graph_balancer.ricci import Ricci
from util.Functions import get_from_nested_dict
from util.Types import *


def get_balancer(config: ConfigDict) -> GraphBalancer:
    balancer_algorithm = get_from_nested_dict(
        config, list_of_keys=["graph_balancer", "algorithm"], raise_error=True).lower()
    balancer = get_balancer_algorithm(balancer_algorithm, config)
    return GraphBalancer(balancer)


def get_balancer_algorithm(name: str, config) -> AbstractGraphBalancer:
    if name == "ricci":
        return Ricci(config)
    elif name == "random":
        return RandomGraphBalancer(config)
    elif name == "none":
        return None
    else:
        raise NotImplementedError("Implement your balancing algorithms here!")
