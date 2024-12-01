# ai libraries
import torch
import torch.nn.functional as F


class CAM:
    """
    This class computes the CAMs for a atch of images.

    Attr:
        model: model used to classify.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        This function is the constructor of the CAM class.

        Args:
            model: model used to classify.

        Returns:
            None.
        """

        self.model = model

    def explain(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This function computes the class activation maps (CAMs) for a
        given input tensor, which are used to visualize the
        discriminative regions in the input image that contribute
        most to the model's prediction.

        Args:
            inputs: input tensor representing the input images.
                Dimensions: [batch_size, channels, height, width].

        Returns:
            CAM tensor of shape representing the class activation maps
                for each input image. Dimensions: [batch_size, height,
                width].
        """

        # TODO
        cnn_out = self.model.cnn_net(inputs)
        gap_out = self.model.gap(cnn_out)
        gap_out = torch.flatten(gap_out, 1)
        class_out = self.model.classifier(gap_out)
        max_class = torch.argmax(class_out, dim=1)
        lista_params = self.model.classifier.weight
        weights = lista_params[max_class]
        return torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * cnn_out, dim=1)


class GradCAM:
    """
    This class computes the GradCAMs for a atch of images.

    Attr:
        model: model used to classify.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        This function is the constructor of the GradCAM class.

        Args:
            model: model used to classify.

        Returns:
            None.
        """

        self.model = model

    def explain(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This function computes the GradCAMs for a batch of images.

        Args:
            inputs: input tensor representing the input images.
                Dimensions: [batch_size, channels, height, width].

        Returns:
            GradCAM tensor of shape representing the class activation
                maps for each input image. Dimensions: [batch_size,
                height, width].
        """

        # TODO

        cnn_out = self.model.cnn_net(inputs)
        gap_out = self.model.gap(cnn_out)
        gap_out = torch.flatten(gap_out, 1)
        class_out = self.model.classifier(gap_out)
        max_class = torch.argmax(class_out, dim=1)

        lista_params = self.model.classifier.weight
        weights = lista_params[max_class]
        grad_cnn = weights.unsqueeze(-1).unsqueeze(-1) / (
            cnn_out.shape[-2] * cnn_out.shape[-1]
        )

        return torch.sum(cnn_out * grad_cnn, dim=1)
