from typing import Optional, Tuple


from pathlib import Path
import cv2

import numpy as np

import torch

from PIL import Image



from src.utils.logger import logger





class GradCAM:

    """

    Grad-CAM implementation for CNN-based image classification.



    Features

    --------

    ✔ Grad-CAM Heatmap

    ✔ Heatmap Overlay

    ✔ Automatic Normalization

    ✔ Anatomical Region Detection

    ✔ Save Heatmap

    ✔ Save Overlay

    """



    def __init__(

        self,

        model: torch.nn.Module,

        target_layer: torch.nn.Module,

    ) -> None:



        logger.info("=" * 60)

        logger.info("Initializing Grad-CAM")

        logger.info("=" * 60)



        self.model = model

        self.target_layer = target_layer



        self.gradients = None

        self.activations = None



        self.forward_handle = self.target_layer.register_forward_hook(

            self._save_activations

        )



        self.backward_handle = self.target_layer.register_full_backward_hook(

            self._save_gradients

        )



        logger.info("Grad-CAM hooks registered successfully.")



    def _save_activations(

        self,

        module,

        inputs,

        output,

    ):

        self.activations = output.detach()



    def _save_gradients(

        self,

        module,

        grad_input,

        grad_output,

    ):

        self.gradients = grad_output[0].detach()



    def remove_hooks(self):



        if self.forward_handle:

            self.forward_handle.remove()
            self.forward_handle = None



        if self.backward_handle:

            self.backward_handle.remove()
            self.backward_handle = None

    def generate(

        self,

        input_tensor: torch.Tensor,

        target_class: Optional[int] = None,

    ) -> Tuple[np.ndarray, str]:

        """

        Generates Grad-CAM heatmap.



        Returns

        -------

        heatmap : numpy.ndarray

            Normalized heatmap (0-1)



        region : str

            Activated anatomical region

        """



        logger.info("Generating Grad-CAM...")



        try:



            self.model.eval()



            # Clear previous tensors

            self.gradients = None

            self.activations = None



            device = next(self.model.parameters()).device

            input_tensor = input_tensor.to(device)



            # Forward

            output = self.model(input_tensor)



            if target_class is None:

                target_class = torch.argmax(output, dim=1).item()



            logger.info(f"Target Class : {target_class}")



            # Backward

            self.model.zero_grad()



            score = output[:, target_class]



            score.backward()



            if self.gradients is None:

                raise RuntimeError("Gradients not captured.")



            if self.activations is None:

                raise RuntimeError("Activations not captured.")



            gradients = self.gradients[0].cpu().numpy()

            activations = self.activations[0].cpu().numpy()



            # Channel weights

            weights = np.mean(gradients, axis=(1, 2))



            cam = np.zeros(

                activations.shape[1:],

                dtype=np.float32,

            )



            for weight, activation in zip(weights, activations):

                cam += weight * activation



            # ReLU

            cam = np.maximum(cam, 0)



            # Resize to original input size

            cam = cv2.resize(

                cam,

                (

                    input_tensor.shape[-1],

                    input_tensor.shape[-2],

                ),

                interpolation=cv2.INTER_LINEAR,

            )



            # Normalize

            cam -= cam.min()



            if cam.max() != 0:

                cam /= cam.max()



            region = self._determine_heatmap_region(cam)



            logger.info(

                f"Grad-CAM generated successfully. Region: {region}"

            )



            return cam, region



        except Exception as e:



            logger.exception("Grad-CAM generation failed.")



            raise RuntimeError(

                "Unable to generate Grad-CAM."

            ) from e

    def overlay_heatmap(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> Image.Image:
        """
        Overlay Grad-CAM heatmap onto the original MRI image.

        Returns:
            PIL.Image.Image
        """

        if not isinstance(original_image, Image.Image):
            raise TypeError(
                "original_image must be a PIL.Image.Image."
            )

        image = np.array(original_image)

        heatmap = cv2.resize(
            heatmap,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET,
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB,
        )

        overlay = cv2.addWeighted(
            image,
            1 - alpha,
            heatmap,
            alpha,
            0,
        )

        return Image.fromarray(overlay)

    def _determine_heatmap_region(

        self,

        cam: np.ndarray,

    ) -> str:

        """

        Determine which anatomical region has the

        strongest Grad-CAM activation.

        """



        height, width = cam.shape



        left_score = np.sum(cam[:, : width // 2])

        right_score = np.sum(cam[:, width // 2:])



        top_score = np.sum(cam[: height // 2, :])

        bottom_score = np.sum(cam[height // 2:, :])



        vertical = (

            "Superior / Frontal"

            if top_score >= bottom_score

            else "Inferior / Occipital"

        )



        horizontal = (

            "Left Hemisphere"

            if left_score >= right_score

            else "Right Hemisphere"

        )



        return f"{vertical} | {horizontal}"





    def __del__(self):

        """

        Remove hooks automatically.

        """



        try:

            self.remove_hooks()

        except Exception:

            pass