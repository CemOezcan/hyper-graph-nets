import torch

from util.Types import *
import torch.nn.functional as F
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from util.pytorch.Feedforward import Feedforward
from torch.utils.data import DataLoader
from util.pytorch.TorchUtil import detach
import numpy as np


class BinaryClassifier(AbstractIterativeAlgorithm):
    def __init__(self, config: ConfigDict) -> None:
        super().__init__(config=config)
        self._network_config = config.get("algorithm").get("network")

        self._network = None
        self._optimizer = None

        self.loss_function = F.binary_cross_entropy_with_logits
        self._learning_rate = self._network_config.get("learning_rate")

    def initialize(self, task_information: ConfigDict) -> None:
        import torch.optim as optim

        self._network = Feedforward(feedforward_config=self._network_config.get("feedforward"),
                    in_features=task_information.get("input_dimension"),
                    out_features=1)
        self._optimizer = optim.Adam(self._network.parameters(), lr=self._learning_rate)

    def score(self, inputs: np.ndarray, labels: np.ndarray) -> ScalarDict:
        num_samples = len(inputs)
        with torch.no_grad():
            inputs = torch.Tensor(inputs)
            labels = torch.Tensor(labels)
            self._network.eval()
            predictions = self._network(inputs)
            predictions = predictions.squeeze()
            loss = self.loss_function(predictions, labels)
            loss = loss.item()
            accuracy = detach(sum(predictions[labels == 1] >= 0.5) + sum(predictions[labels == 0] < 0.5))

        accuracy = accuracy / num_samples

        return {"accuracy": accuracy,
                "loss": loss
                }

    def predict(self, samples: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(samples, np.ndarray):
            samples = torch.Tensor(samples.astype(np.float32))
        evaluations = self._network(samples)
        return detach(evaluations)

    def fit_iteration(self, train_dataloader: DataLoader) -> None:
        self._network.train()
        for data in train_dataloader:  # for each batch
            self._run_minibatch(data=data)

    def _run_minibatch(self, data: tuple) -> float:
        inputs, labels = data  # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients
        self._optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self._network(inputs)
        outputs = outputs.squeeze()
        loss = self.loss_function(outputs, labels)
        loss.backward()
        self._optimizer.step()
        return loss.item()

    @property
    def network(self):
        return self._network
