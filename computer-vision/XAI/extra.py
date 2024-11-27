# ai libraries
import torch
import torch.nn.functional as F

# other libraries
import copy
from typing import Callable


class IntegratedGradients:
    """
    This is the class for computing integrated gradients.

    Attr:
        model: model used to classify.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        This function is the constructor

        Args:
            model: model used to classify.

        Returns:
            None.
        """

        self.model = model

    def explain(self, inputs: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """
        This method computes the explanation.

        Args:
            inputs: inputs tensor. Dimensions: [batch, channels,
                height, width].
            steps: number of steps.

        Raises:
            RuntimeError: No gradients were computed during the
                backward pass.

        Returns:
            saliency maps tensor. Dimensions: [batch, height, width].
        """

        # TODO
        # inputs.requires_grad = True
        baseline = torch.zeros_like(inputs)
        resta = inputs - baseline
        saliency_map = torch.zeros_like(inputs)
        for i in range(steps + 1):
            scaled_inputs = baseline + i / steps * resta
            scaled_inputs.requires_grad = True
            outputs = self.model(scaled_inputs)
            logits, _ = torch.max(outputs, dim=1)
            torch.sum(logits).backward()
            if scaled_inputs.grad is not None:
                saliency_map += scaled_inputs.grad
        return torch.abs(resta * saliency_map / steps).sum(dim=1)


class GuidedBackprop:
    @torch.no_grad()
    def __init__(self, model: torch.nn.Module) -> None:
        """
        This is the constructor for GuidedBackpropSaliencyMap:.

        Args:
            model: model used to classify.
        """

        # set attributes
        self.model = copy.deepcopy(model)
        self.register_hooks()

    def register_hooks(self) -> None:
        """
        This function registers the hooks needed for GuidedBackpropSaliencyMap(SaliencyMap):.

        Returns:
            None.
        """

        # TODO
        self.model.cnn_net[1]
        for child in self.model.cnn_net:
            if child.__class__.__name__ == "ReLU":
                child.register_backward_hook(
                    lambda _, i, o: tuple([torch.clamp(grad, min=0) for grad in i])
                )
            elif child.__class__.__name__ == "Block":
                for grandchild in child.net:
                    if grandchild.__class__.__name__ == "ReLU":
                        grandchild.register_backward_hook(
                            lambda _, i, o: tuple(
                                [torch.clamp(grad, min=0) for grad in i]
                            )
                        )

    def explain(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the explanation.

        Args:
            inputs: inputs tensor. Dimensions: [batch, channels,
                height, width].

        Raises:
            RuntimeError: No gradients were computed during the
                backward pass.

        Returns:
            saliency maps tensor. Dimensions: [batch, height, width].
        """

        # TODO
        inputs.requires_grad = True
        outputs = self.model(inputs)
        logits, _ = torch.max(outputs, dim=1)
        torch.sum(logits).backward()
        if inputs.grad is not None:
            return torch.max(torch.abs(inputs.grad), dim=1).values
        return None
