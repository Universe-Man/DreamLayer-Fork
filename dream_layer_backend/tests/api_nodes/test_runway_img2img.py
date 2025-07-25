import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import requests
import base64
import io
from PIL import Image

# Import the Runway integration from dream_layer_backend_utils
from dream_layer_backend.dream_layer_backend_utils.runway_img2img import (
    RunwayImg2ImgNode,
)


class TestRunwayImg2ImgNode:
    """Test suite for Runway Gen-4 Image-to-Image Node with mocked HTTP requests."""

    @pytest.fixture
    def node(self):
        """Create a node instance for testing."""
        return RunwayImg2ImgNode()

    @pytest.fixture
    def sample_image_tensor(self):
        """Create a sample image tensor for testing."""
        # Create a 64x64 RGB image tensor in format [1, height, width, channels]
        return torch.rand(1, 64, 64, 3)

    @pytest.fixture
    def mock_api_response(self):
        """Create a mock API response."""
        # Create a simple test image as base64
        test_image = Image.new("RGB", (64, 64), color="red")
        buffer = io.BytesIO()
        test_image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"image": f"data:image/png;base64,{image_b64}", "status": "completed"}

    def test_input_types_structure(self, node):
        """Test that INPUT_TYPES returns the correct structure."""
        input_types = node.INPUT_TYPES()

        assert "required" in input_types
        assert "optional" in input_types

        # Check required inputs
        required = input_types["required"]
        assert "image" in required
        assert "prompt" in required
        assert required["image"] == ("IMAGE",)
        assert required["prompt"][0] == "STRING"

        # Check optional inputs
        optional = input_types["optional"]
        assert "timeout" in optional
        assert optional["timeout"][0] == "INT"
        assert optional["timeout"][1]["default"] == 120

    def test_return_types(self, node):
        """Test that return types are correctly defined."""
        assert node.RETURN_TYPES == ("IMAGE",)
        assert node.RETURN_NAMES == ("image",)
        assert node.FUNCTION == "generate_image"
        assert node.CATEGORY == "image/ai"

    def test_missing_api_key_error(self, node, sample_image_tensor):
        """Test that missing RUNWAY_API_KEY raises a helpful error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                node.generate_image(sample_image_tensor, "test prompt")

            error_message = str(exc_info.value)
            assert "RUNWAY_API_KEY" in error_message
            assert "environment variable is required" in error_message
            assert "runwayml.com" in error_message

    def test_tensor_to_base64_conversion(self, node, sample_image_tensor):
        """Test tensor to base64 conversion."""
        base64_result = node._tensor_to_base64(sample_image_tensor)

        assert base64_result.startswith("data:image/png;base64,")
        assert len(base64_result) > 50  # Should be a substantial base64 string

    def test_base64_to_tensor_conversion(self, node):
        """Test base64 to tensor conversion."""
        # Create a test image
        test_image = Image.new("RGB", (32, 32), color="blue")
        buffer = io.BytesIO()
        test_image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        full_b64 = f"data:image/png;base64,{image_b64}"

        result_tensor = node._base64_to_tensor(full_b64)

        assert isinstance(result_tensor, torch.Tensor)
        assert result_tensor.shape == (1, 32, 32, 3)  # [batch, height, width, channels]
        assert result_tensor.min() >= 0.0
        assert result_tensor.max() <= 1.0

    @patch("requests.post")
    @patch.dict("os.environ", {"RUNWAY_API_KEY": "test-api-key"})
    def test_successful_api_call(
        self, mock_post, node, sample_image_tensor, mock_api_response
    ):
        """Test successful API call with mocked HTTP response."""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_api_response
        mock_post.return_value = mock_response

        # Call the function
        result = node.generate_image(sample_image_tensor, "a beautiful landscape")

        # Verify the request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args

        # Check URL
        assert call_args[0][0] == "https://api.runwayml.com/v1/text_to_image"

        # Check headers
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["X-Runway-Version"] == "2024-11-06"
        assert headers["Content-Type"] == "application/json"

        # Check payload
        payload = call_args[1]["json"]
        assert payload["prompt"] == "a beautiful landscape"
        assert "promptImage" in payload
        assert payload["promptImage"].startswith("data:image/png;base64,")

        # Check timeout
        assert call_args[1]["timeout"] == 120

        # Verify result
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)

    @patch("requests.post")
    @patch.dict("os.environ", {"RUNWAY_API_KEY": "test-api-key"})
    def test_api_error_handling(self, mock_post, node, sample_image_tensor):
        """Test API error handling."""
        # Mock a failed response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request: Invalid prompt"
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError) as exc_info:
            node.generate_image(sample_image_tensor, "test prompt")

        error_message = str(exc_info.value)
        assert "Runway API error (400)" in error_message
        assert "Bad Request: Invalid prompt" in error_message

    @patch("requests.post")
    @patch.dict("os.environ", {"RUNWAY_API_KEY": "test-api-key"})
    def test_timeout_handling(self, mock_post, node, sample_image_tensor):
        """Test timeout handling."""
        # Mock a timeout
        mock_post.side_effect = requests.exceptions.Timeout()

        with pytest.raises(RuntimeError) as exc_info:
            node.generate_image(sample_image_tensor, "test prompt", timeout=30)

        error_message = str(exc_info.value)
        assert "Request timed out after 30 seconds" in error_message

    @patch("requests.post")
    @patch.dict("os.environ", {"RUNWAY_API_KEY": "test-api-key"})
    def test_connection_error_handling(self, mock_post, node, sample_image_tensor):
        """Test connection error handling."""
        # Mock a connection error
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        with pytest.raises(RuntimeError) as exc_info:
            node.generate_image(sample_image_tensor, "test prompt")

        error_message = str(exc_info.value)
        assert "Request failed: Connection failed" in error_message

    @patch("requests.post")
    @patch.dict("os.environ", {"RUNWAY_API_KEY": "test-api-key"})
    def test_custom_timeout_parameter(
        self, mock_post, node, sample_image_tensor, mock_api_response
    ):
        """Test that custom timeout parameter is used."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_api_response
        mock_post.return_value = mock_response

        # Call with custom timeout
        node.generate_image(sample_image_tensor, "test prompt", timeout=60)

        # Verify timeout was passed correctly
        call_args = mock_post.call_args
        assert call_args[1]["timeout"] == 60

    @patch("requests.post")
    @patch.dict("os.environ", {"RUNWAY_API_KEY": "test-api-key"})
    def test_alternative_response_format(self, mock_post, node, sample_image_tensor):
        """Test handling of alternative API response format."""
        # Create test image
        test_image = Image.new("RGB", (64, 64), color="green")
        buffer = io.BytesIO()
        test_image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Mock response with 'data' array format
        alternative_response = {
            "data": [
                {"image": f"data:image/png;base64,{image_b64}", "id": "generated-123"}
            ],
            "status": "completed",
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = alternative_response
        mock_post.return_value = mock_response

        result = node.generate_image(sample_image_tensor, "test prompt")

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)

    def test_docstring_completeness(self, node):
        """Test that the node has comprehensive documentation."""
        docstring = node.__class__.__doc__

        assert docstring is not None
        assert "RUNWAY_API_KEY" in docstring
        assert "timeout" in docstring
        assert "image" in docstring
        assert "prompt" in docstring

        # Check that generate_image method has documentation
        method_doc = node.generate_image.__doc__
        assert method_doc is not None
        assert "Args:" in method_doc or "Parameters:" in method_doc
        assert "Returns:" in method_doc
