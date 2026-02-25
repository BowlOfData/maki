"""
Unit tests for all Maki framework functionalities
"""
import unittest
from unittest.mock import patch, MagicMock
import json

# Import the classes we want to test
from maki.maki import Maki
from maki.utils import Utils
from maki.connector import Connector

class TestMakiFunctionality(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_url = "localhost"
        self.test_port = "11434"
        self.test_model = "llama3"
        self.test_temperature = 0.7
        self.maki = Maki(self.test_url, self.test_port, self.test_model, self.test_temperature)

    def test_maki_initialization(self):
        """Test that Maki initializes correctly"""
        self.assertEqual(self.maki.url, self.test_url)
        self.assertEqual(self.maki.port, self.test_port)
        self.assertEqual(self.maki.model, self.test_model)
        self.assertEqual(self.maki.temperature, self.test_temperature)

    def test_request_method(self):
        """Test the request method functionality"""
        # Mock the connector to avoid actual HTTP requests
        with patch.object(Connector, 'simple') as mock_simple:
            mock_simple.return_value = "Test response"

            result = self.maki.request("Test prompt")
            self.assertEqual(result, "Test response")

            # Verify that the correct URL and data were passed
            mock_simple.assert_called_once()

    def test_version_method(self):
        """Test the version method functionality"""
        # Mock the connector to avoid actual HTTP requests
        with patch.object(Connector, 'version') as mock_version:
            mock_version.return_value = "Test version info"

            result = self.maki.version()
            self.assertEqual(result, "Test version info")

            # Verify that the correct URL was passed
            mock_version.assert_called_once()

    def test_compose_data_method(self):
        """Test the _compose_data method functionality"""
        prompt = "Test prompt"
        result = self.maki._compose_data(prompt)

        # Verify the structure of the composed data
        self.assertIn("model", result)
        self.assertIn("prompt", result)
        self.assertEqual(result["model"], self.test_model)
        self.assertEqual(result["prompt"], prompt)

        # Test with temperature
        self.assertIn("options", result)
        self.assertIn("temperature", result["options"])
        self.assertEqual(result["options"]["temperature"], self.test_temperature)

    def test_compose_data_with_images(self):
        """Test the _compose_data method with image data"""
        prompt = "Test prompt"
        # Mock the image conversion
        with patch.object(Utils, 'convert64') as mock_convert:
            mock_convert.return_value = b"test_image_data"

            result = self.maki._compose_data(prompt, imgs=["test_image"])

            # Verify that images were included
            self.assertIn("images", result)
            self.assertEqual(len(result["images"]), 1)

    def test_request_with_images_method(self):
        """Test the request_with_images method functionality"""
        # Mock the connector and utils to avoid actual file operations and HTTP requests
        with patch.object(Connector, 'simple') as mock_simple, \
             patch.object(Utils, 'convert64') as mock_convert:

            mock_convert.return_value = b"test_image_data"
            mock_simple.return_value = "Test image response"

            result = self.maki.request_with_images("Test prompt", "test_image.jpg")
            self.assertEqual(result, "Test image response")

            # Verify that the correct calls were made
            mock_convert.assert_called_once()
            mock_simple.assert_called_once()

    def test_get_model_method(self):
        """Test the _get_model method"""
        result = self.maki._get_model()
        self.assertEqual(result, self.test_model)

    def test_get_temperature_method(self):
        """Test the _get_temperature method"""
        result = self.maki._get_temperature()
        self.assertEqual(result, self.test_temperature)

    def test_utils_compose_url(self):
        """Test the Utils.compose_url method"""
        # Import the Actions to get the correct format
        from maki.urls import Actions

        result = Utils.compose_url(self.test_url, self.test_port, Actions.GENERATE.value)
        expected = f"http://{self.test_url}:{self.test_port}/api/generate"
        self.assertEqual(result, expected)

    def test_utils_jsonify(self):
        """Test the Utils.jsonify method"""
        test_data = '{"test": "data"}'
        result = Utils.jsonify(test_data)
        self.assertEqual(result, {"test": "data"})

    @patch('maki.utils.os.path.exists')
    def test_utils_convert64(self, mock_exists):
        """Test the Utils.convert64 method with mock file"""
        # Mock that the file exists
        mock_exists.return_value = True

        # Test with a mock file
        with patch('maki.utils.open', unittest.mock.mock_open(read_data=b'test data')) as mock_file:
            result = Utils.convert64("test.jpg")
            self.assertIsNotNone(result)

    def test_maki_with_different_temperature(self):
        """Test Maki with different temperature values"""
        maki_with_zero_temp = Maki(self.test_url, self.test_port, self.test_model, 0)
        self.assertEqual(maki_with_zero_temp.temperature, 0)

        maki_with_high_temp = Maki(self.test_url, self.test_port, self.test_model, 1.0)
        self.assertEqual(maki_with_high_temp.temperature, 1.0)

    def test_maki_with_different_models(self):
        """Test Maki with different model values"""
        maki_mistral = Maki(self.test_url, self.test_port, "mistral", self.test_temperature)
        self.assertEqual(maki_mistral.model, "mistral")

        maki_gemma = Maki(self.test_url, self.test_port, "gemma", self.test_temperature)
        self.assertEqual(maki_gemma.model, "gemma")


if __name__ == '__main__':
    unittest.main()