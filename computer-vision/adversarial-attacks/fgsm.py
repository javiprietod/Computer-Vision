import torch
from src.utils import visualize_perturbations


class FastGradientSignMethod:
    """
    This class implements the white-box adversarial attack Fast Gradient Sign Method
    (FGSM): x' = x + epsilon * dL/dx.
    """

    def __init__(self, model: torch.nn.Module, loss: torch.nn.Module) -> None:
        """
        Constructor of the class.

        Parameters
        ----------
        model : Model used.
        loss  : Loss used by the model.
        """

        self.model = model
        self.loss = loss

    def _get_perturbations(self, img: torch.Tensor, label: int) -> torch.Tensor:
        """
        Obtains the gradient of the loss with respect to the input.

        Parameters
        ----------
        img   : Original image. Dimensions: [channels, height, width].
        label : Real label of the image.

        Returns
        -------
        Perturbation for the image. Dimensions: [channels, height, width].
        """

        # TODO
        img.requires_grad = True
        self.model.zero_grad()
        output = self.model(img.unsqueeze(0))
        max_class = output.argmax()
        if max_class == label:
            loss = self.loss(output, torch.tensor([label]))
            loss.backward()
            if img.grad is not None:
                sign = img.grad.data.sign()
            img.requires_grad = False
            return sign
        return torch.zeros_like(img)

    def perturb_img(
        self,
        img: torch.Tensor,
        label: int,
        epsilon: float = 1e-2,
        show: bool = True,
        title: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perturbs an image.

        Parameters
        ----------
        img     : Original image.
        label   : Real label of the image.
        epsilon : Epsilon parameter.
        show    : Boolean parameter that decides whether to save the figure or not.
        title   : Title of the image in case it is saved.

        Returns
        -------
        perturbed_img : Perturbed image. Dimensions: [channels, height, width].
        perturbations : Perturbations made. Dimensions: [channels, height, width].
        """

        # TODO
        perturbations = epsilon * self._get_perturbations(img, label)
        perturbed_img = img + perturbations
        perturbed_img = torch.clamp(perturbed_img, 0, 1)
        if show:
            visualize_perturbations(perturbed_img, img, label, self.model, title)
        return perturbed_img, perturbations
