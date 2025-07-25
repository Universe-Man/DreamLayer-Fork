"""
Runway Gen-4 Image-to-Image Integration

Handles API calls to Runway's reference-to-image endpoint for img2img generation.
This utility integrates with the DreamLayer workflow system.
"""

import os
import requests
import torch
import numpy as np
from PIL import Image
import io
import base64
import logging
from typing import Optional, Dict, Any, Tuple

# Import your existing utilities
from .api_key_injector import read_api_keys_from_env

logger = logging.getLogger(__name__)


class RunwayImg2ImgIntegration:
    """
    Runway Gen-4 Image-to-Image API Integration

    Integrates with Runway's reference-to-image endpoint to generate images
    based on an input reference image and text prompt.

    Required Environment Variables:
        RUNWAY_API_KEY: Your Runway API key for authentication

    Usage:
        integration = RunwayImg2ImgIntegration()
        result = integration.generate_image(image_tensor, "a beautiful landscape")
    """

    def __init__(self):
        self.api_url = "https://api.runwayml.com/v1/text_to_image"
        self.api_version = "2024-11-06"
        self.api_key_name = "RUNWAY_API_KEY"

    def validate_api_key(self) -> str:
        """
        Validate and return the Runway API key.

        Returns:
            str: The API key

        Raises:
            ValueError: If API key is missing or invalid
        """
        # Use existing api_key_injector system
        all_api_keys = read_api_keys_from_env()
        api_key = all_api_keys.get(self.api_key_name)

        if not api_key:
            raise ValueError(
                f"{self.api_key_name} environment variable is required. "
                "Please set your Runway API key in the .env file in your project root. "
                "You can get your API key from https://runwayml.com/"
            )

        return api_key

    def prepare_image_for_api(self, image_data: Any) -> str:
        """
        Convert image data to base64 format for API submission.

        Args:
            image_data: Image tensor, PIL Image, or numpy array

        Returns:
            str: Base64 encoded image with data URL format
        """
        # Handle different input types
        if isinstance(image_data, torch.Tensor):
            image_array = self._tensor_to_array(image_data)
        elif isinstance(image_data, np.ndarray):
            image_array = image_data
        elif isinstance(image_data, Image.Image):
            image_array = np.array(image_data)
        else:
            raise ValueError(f"Unsupported image type: {type(image_data)}")

        # Ensure uint8 format
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)

        # Convert to PIL Image
        pil_image = Image.fromarray(image_array)

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{image_base64}"

    def _tensor_to_array(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array, handling batch dimensions."""
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # Remove batch dimension

        return tensor.cpu().numpy()

    def parse_api_response(self, response_data: Dict[str, Any]) -> str:
        """
        Parse the API response to extract the generated image.

        Args:
            response_data: JSON response from Runway API

        Returns:
            str: Base64 encoded generated image

        Raises:
            ValueError: If response format is unexpected
        """
        # Handle different possible response formats
        if "image" in response_data:
            return response_data["image"]
        elif "data" in response_data and len(response_data["data"]) > 0:
            return response_data["data"][0]["image"]
        elif "output" in response_data:
            return response_data["output"]
        else:
            logger.error(f"Unexpected API response format: {response_data}")
            raise ValueError("Unexpected API response format - no image found")

    def convert_response_to_tensor(self, base64_image: str) -> torch.Tensor:
        """
        Convert base64 image response back to tensor format.

        Args:
            base64_image: Base64 encoded image from API

        Returns:
            torch.Tensor: Image tensor in format [1, height, width, channels]
        """
        # Remove data URL prefix if present
        if base64_image.startswith("data:image"):
            base64_image = base64_image.split(",")[1]

        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_image)

        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if not already
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Convert to numpy array and normalize to 0-1 range
        image_array = np.array(pil_image).astype(np.float32) / 255.0

        # Convert to tensor with batch dimension [1, height, width, channels]
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)

        return image_tensor

    def generate_image(
        self, input_image: Any, prompt: str, timeout: int = 120, **kwargs
    ) -> torch.Tensor:
        """
        Generate an image using Runway's reference-to-image endpoint.

        Args:
            input_image: Input image (tensor, PIL Image, or numpy array)
            prompt: Text prompt for generation
            timeout: Request timeout in seconds
            **kwargs: Additional parameters (for future extensibility)

        Returns:
            torch.Tensor: Generated image tensor

        Raises:
            ValueError: If API key is missing or parameters are invalid
            RuntimeError: If API request fails
        """
        try:
            # Validate API key
            api_key = self.validate_api_key()

            # Prepare image for API
            prompt_image_b64 = self.prepare_image_for_api(input_image)

            # Prepare headers
            headers = {
                "Authorization": f"Bearer {api_key}",
                "X-Runway-Version": self.api_version,
                "Content-Type": "application/json",
            }

            # Prepare payload
            payload = {"prompt": prompt, "promptImage": prompt_image_b64}

            # Add any additional parameters from kwargs
            if kwargs:
                payload.update(kwargs)

            logger.info(f"Making Runway API request with prompt: {prompt[:50]}...")

            # Make API request
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=timeout
            )

            # Handle response
            if response.status_code != 200:
                error_msg = (
                    f"Runway API error ({response.status_code}): {response.text}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            result = response.json()
            logger.info("Runway API request successful")

            # Parse response and convert back to tensor
            generated_image_b64 = self.parse_api_response(result)
            output_tensor = self.convert_response_to_tensor(generated_image_b64)

            return output_tensor

        except requests.exceptions.Timeout:
            error_msg = f"Request timed out after {timeout} seconds"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)


# Convenience function for workflow integration
def runway_img2img(image, prompt: str, timeout: int = 120, **kwargs) -> torch.Tensor:
    """
    Convenience function for Runway image-to-image generation.

    This function can be called directly from workflow execution scripts.

    Args:
        image: Input image (various formats supported)
        prompt: Text prompt for generation
        timeout: Request timeout in seconds
        **kwargs: Additional parameters

    Returns:
        torch.Tensor: Generated image tensor
    """
    integration = RunwayImg2ImgIntegration()
    return integration.generate_image(image, prompt, timeout, **kwargs)


# For ComfyUI node compatibility (if needed)
class RunwayImg2ImgNode:
    """
    ComfyUI-compatible node wrapper for Runway integration.

    This provides the interface expected by ComfyUI while using
    the underlying RunwayImg2ImgIntegration utility.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "A beautiful landscape"},
                ),
            },
            "optional": {
                "timeout": ("INT", {"default": 120, "min": 30, "max": 300, "step": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "image/ai"

    def __init__(self):
        self.integration = RunwayImg2ImgIntegration()

    def generate_image(
        self, image: torch.Tensor, prompt: str, timeout: int = 120
    ) -> Tuple[torch.Tensor]:
        """ComfyUI node interface."""
        result = self.integration.generate_image(image, prompt, timeout)
        return (result,)


# Export the main classes and functions
__all__ = ["RunwayImg2ImgIntegration", "RunwayImg2ImgNode", "runway_img2img"]
