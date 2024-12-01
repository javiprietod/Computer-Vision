import torch
import torch.nn as nn
from torch._tensor import Tensor
import copy
from typing import Callable, Optional, Sequence, Tuple


class CascadeChecker:
    def __init__(self, saliency_map_creator) -> None:
        """
        Constructor for CascadeChecker.

        Args:
            saliency_map_creator: object used for saliency generation
        """

        self.saliency_map_creator = saliency_map_creator
        self.baseline_saliency_map_creator = copy.deepcopy(saliency_map_creator)

        self.original_model_convs: list[nn.Module] = [
            i
            for i in self.saliency_map_creator.model.modules()
            if type(i) == torch.nn.Conv2d
        ]
        self.randomized_model_convs: list[nn.Module] = [
            i
            for i in self.baseline_saliency_map_creator.model.modules()
            if type(i) == torch.nn.Conv2d
        ]
        # noisy method may have multiple input gradients per conv
        self.original_model_grads: list[list[Tensor]] = []
        self.randomized_model_grads: list[list[Tensor]] = []

        return None

    def __len__(self) -> int:
        """
        Returns:
            int: integer indicating the number of convolutional layers in the models
        """
        conv_steps: int = len(self.original_model_convs)
        return conv_steps

    @property
    def convolutional_layers(self) -> Tuple[list[nn.Module], list[nn.Module]]:
        """
        Returns:
            list[nn.Module]: list with the convolutional layers in the original model
            list[nn.Module]: list with the convolutional layers in the randomized model
        """
        return (self.original_model_convs, self.randomized_model_convs)

    @property
    def feature_maps_gradients(self) -> Tuple[list[list[Tensor]], list[list[Tensor]]]:
        """
        Returns:
            list[list[Tensor]]: convolutional input grads in the original model
            list[list[Tensor]]: convolutional input grads in the randomized model
        """
        return (self.original_model_grads, self.randomized_model_grads)

    def sanity_check(
        self,
        inputs: Tensor,
        idx: int,
        metric: Callable[[Tensor, Tensor], Tensor],
        args: Optional[Sequence] = [],
    ) -> Tensor:
        """
        Performs a sanity check by comparing the gradients of internal feature maps
        between the trained model and a randomized baseline model.

        This method evaluates the similarity between the gradients of the internal
        feature maps of two models:
        1. *Trained Model*: The original model provided during initialization.
        2. *Baseline Model*: A copy of the trained model, which is partially
        randomized starting from a specified convolutional layer index.

        Args:
            inputs (Tensor):
                A tensor containing the input data to be fed into the models.
                Shape: (batch_size, channels, height, width).

            idx (int):
                The index of the convolutional layer from which to start randomizing
                the baseline model. All convolutional layers at and after this index
                will have their parameters reset to random values.

            metric (Callable[[Tensor, Tensor], Tensor]):
                A function that takes two tensors as input and returns a tensor
                representing the similarity between them. This function is used to
                compute the similarity between the gradients of the trained and baseline
                models' feature maps.

            args (Optional[Sequence], optional):
                Additional positional arguments to be passed to the explain method of
                the saliency generators. Defaults to an empty list.

        Returns:
            Tensor:
                A tensor containing similarity scores for each convolutional layer
                between the trained and baseline models. The shape of the tensor is
                (batch_size, number_of_convolutional_layers), where each element [i, j]
                represents the similarity score for the i-th input in the batch and the
                j-th convolutional layer.
        """

        similarities: Tensor = torch.empty(self.__len__(), inputs.shape[0])
        trained_saliency_maps = inputs
        baseline_saliency_maps = inputs
        for j in range(self.__len__()):
            if j >= idx: # if using main method, this could be j == idx
                self.randomized_model_convs[j].reset_parameters()
            trained_saliency_maps: Tensor = self.original_model_convs[j].forward(
                trained_saliency_maps
            )
            baseline_saliency_maps: Tensor = self.randomized_model_convs[j].forward(
                baseline_saliency_maps
            )
            similarities[j] = torch.mean(
                metric(trained_saliency_maps, baseline_saliency_maps)
            )

        return similarities
    
    def main(self, inputs: Tensor, metric: Callable[[Tensor, Tensor], Tensor], args: Optional[Sequence] = []) -> Tensor:
        """
        
            Implement the main method for the CascadeChecker class. This method
            should perform a sanity check from the top convolutional layer to the
            bottom convolutional layer of the model cumulatively. 
        
        """
        similarities: Tensor = torch.empty(self.__len__(), self.__len__(), inputs.shape[0])
        for idx in range(self.__len__() - 1, -1, -1):
            similarities[idx] = self.sanity_check(inputs, idx, metric, args)

        return similarities



class IndependentChecker:
    def __init__(self, saliency_map_creator) -> None:
        """
        Constructor for IndependentChecker.

        Args:
            saliency_map_creator: object used for saliency generation
        """

        self.saliency_map_creator = saliency_map_creator
        self.baseline_saliency_map_creator = copy.deepcopy(saliency_map_creator)

        self.original_model_convs: list[nn.Module] = [
            i
            for i in self.saliency_map_creator.model.modules()
            if type(i) == torch.nn.Conv2d
        ]
        self.randomized_model_convs: list[nn.Module] = [
            i
            for i in self.baseline_saliency_map_creator.model.modules()
            if type(i) == torch.nn.Conv2d
        ]
        # noisy method may have multiple input gradients per conv
        self.original_model_grads: list[list[Tensor]] = []
        self.randomized_model_grads: list[list[Tensor]] = []

        return None

    def __len__(self) -> int:
        """
        Returns:
            int: integer indicating the number of convolutional layers in the models
        """
        conv_steps: int = len(self.original_model_convs)
        return conv_steps

    @property
    def convolutional_layers(self) -> Tuple[list[nn.Module], list[nn.Module]]:
        """
        Returns:
            list[nn.Module]: list with the convolutional layers in the original model
            list[nn.Module]: list with the convolutional layers in the randomized model
        """
        return (self.original_model_convs, self.randomized_model_convs)

    @property
    def feature_maps_gradients(self) -> Tuple[list[list[Tensor]], list[list[Tensor]]]:
        """
        Returns:
            list[list[Tensor]]: convolutional input grads in the original model
            list[list[Tensor]]: convolutional input grads in the randomized model
        """
        return (self.original_model_grads, self.randomized_model_grads)

    def sanity_check(
        self,
        inputs: Tensor,
        idx: int,
        metric: Callable[[Tensor, Tensor], Tensor],
        args: Optional[Sequence] = [],
    ) -> Tensor:
        """
        Args:
            inputs (Tensor):
                Tensor of input images. Shape: (batch_size, channels, height, width).

            idx (int):
                Indicator of the layer to randomize

            metric (Callable[[Tensor, Tensor], Tensor]):
                Metric of similarity between tensors. [B, C, H, W] -> [B, C]

            args (Optional[Sequence], optional):
                Additional arguments for the self.saliency_map_creator.explain method

        Returns:
            Tensor: similarities per batch per conv. Shape: [convs, batch]

        """

        # Randomize the baseline model starting from the specified convolutional layer index

        similarities: Tensor = torch.empty(self.__len__(), inputs.shape[0])
        similarities: Tensor = torch.empty(self.__len__(), inputs.shape[0])
        trained_saliency_maps = inputs
        baseline_saliency_maps = inputs
        for j in range(self.__len__()):
            if j == idx:
                state_dict = self.randomized_model_convs[j].state_dict()
                self.randomized_model_convs[j].reset_parameters()
            trained_saliency_maps: Tensor = self.original_model_convs[j].forward(
                trained_saliency_maps
            )
            baseline_saliency_maps: Tensor = self.randomized_model_convs[j].forward(
                baseline_saliency_maps
            )
            similarities[j] = torch.mean(
                metric(trained_saliency_maps, baseline_saliency_maps)
            )
            if j == idx:
                self.randomized_model_convs[j].load_state_dict(state_dict)

        return similarities

    def main(self, inputs: Tensor, metric: Callable[[Tensor, Tensor], Tensor], args: Optional[Sequence] = []) -> Tensor:
        """
        
            Implement the main method for the IndependentChecker class. This method
            should perform a sanity check from the top convolutional layer to the
            bottom convolutional layer of the model independently. 
        
        """
        similarities: Tensor = torch.empty(self.__len__(), self.__len__(), inputs.shape[0])
        for idx in range(self.__len__() - 1, -1, -1):
            similarities[idx] = self.sanity_check(inputs, idx, metric, args)

        return similarities