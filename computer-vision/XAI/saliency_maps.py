# ai libraries
import torch
import torch.nn.functional as F

# other libraries
import copy
from typing import Callable


class SaliencyMap:
    """
    This is the class for computing saliency maps.

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


class SmoothGradSaliencyMap(torch.nn.Module):
    """
    This is the class for computing smoothgrad saliency maps.

    Attr:
        model: model used to classify.
    """

    def __init__(self) -> None:
        """
        Thi function is the constructor for SmoothGradSaliencyMap.

        Args:
            model: model used to classify.

        Returns:
            None.
        """

        # call super class constructor
        super().__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        model: torch.nn.Module,
        noise_level: float,
        sample_size: int,
    ) -> torch.Tensor:
        return self.explain(inputs, model, noise_level, sample_size)

    @torch.no_grad()
    def explain(
        self,
        inputs: torch.Tensor,
        model: torch.nn.Module,
        noise_level: float,
        sample_size: int,
    ) -> torch.Tensor:
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
        batch, channels, height, width = inputs.shape
        noise = torch.normal(
            mean=0, std=noise_level, size=(batch, sample_size, channels, height, width)
        )

        inputs_with_noise = inputs.unsqueeze(1) + noise

        outputs_matrix = torch.zeros(batch, height, width)
        inputs_with_noise.requires_grad = True
        with torch.enable_grad():
            for i in range(sample_size):
                outputs = model(inputs_with_noise[:, i])

                logits, _ = torch.max(outputs, dim=1)

                torch.sum(logits).backward()
                if inputs_with_noise.grad is not None:
                    outputs_matrix += torch.max(
                        torch.abs(inputs_with_noise.grad[:, i]), dim=1
                    ).values

            return outputs_matrix / sample_size


class DeConvNet:
    @torch.no_grad()
    def __init__(self, model: torch.nn.Module) -> None:
        """
        This is the constructor for DeConvNet.

        Args:
            model: model used to classify.
        """

        # set attributes
        self.model = copy.deepcopy(model)
        self.register_hooks()

    def register_hooks(self) -> None:
        """
        This function registers the hooks needed for deconvnet.

        Returns:
            None.
        """

        # TODO
        self.model.cnn_net[1]
        for child in self.model.cnn_net:
            if child.__class__.__name__ == "ReLU":
                child.register_backward_hook(
                    lambda _, i, o: tuple([torch.clamp(grad, min=0) for grad in o])
                )
            elif child.__class__.__name__ == "Block":
                for grandchild in child.net:
                    if grandchild.__class__.__name__ == "ReLU":
                        grandchild.register_backward_hook(
                            lambda _, i, o: tuple(
                                [torch.clamp(grad, min=0) for grad in o]
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
            return torch.max(inputs.grad, dim=1).values
        return None
