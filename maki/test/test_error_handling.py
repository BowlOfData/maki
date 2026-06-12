"""
Unit tests for error handling and validation in Maki framework
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to Python path so imports work properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maki.connector import Connector
from maki.utils import Utils
from maki.agents import Agent, AgentManager

class TestErrorHandling(unittest.TestCase):

    def test_utils_jsonify_validation(self):
        """Test that Utils.jsonify validates input"""
        # Test valid JSON
        result = Utils.jsonify('{"test": "data"}')
        self.assertEqual(result, {"test": "data"})

        # Test invalid JSON
        with self.assertRaises(ValueError):
            Utils.jsonify("invalid json")

        with self.assertRaises(ValueError):
            Utils.jsonify("")

        with self.assertRaises(ValueError):
            Utils.jsonify(None)

    def test_utils_convert64_validation(self):
        """Test that Utils.convert64 validates input"""
        # Test with non-existent file - we need to make the path within working directory
        # but simulate that the file doesn't exist by patching os.path.exists
        with patch('maki.utils.os.path.exists') as mock_exists:
            mock_exists.return_value = False
            with self.assertRaises(FileNotFoundError):
                Utils.convert64("non_existent_file.jpg")

        # Test invalid file path
        with self.assertRaises(ValueError):
            Utils.convert64("")

        with self.assertRaises(ValueError):
            Utils.convert64(None)

    def test_utils_convert64_accepts_symlinked_directories(self):
        """Regression §1.4: paths under a symlinked directory (e.g. /tmp on
        macOS) were rejected because abspath != realpath. A symlink in the
        ancestry is not an attack; only escaping an allowed base dir is.
        """
        import base64
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            real_dir = os.path.join(tmp, "real")
            os.makedirs(real_dir)
            img_path = os.path.join(real_dir, "pic.png")
            with open(img_path, "wb") as f:
                f.write(b"fake-image-bytes")

            link_dir = os.path.join(tmp, "link")
            os.symlink(real_dir, link_dir)

            # Access through the symlinked directory must succeed.
            result = Utils.convert64(os.path.join(link_dir, "pic.png"))
            self.assertIsInstance(result, str)
            self.assertEqual(base64.b64decode(result), b"fake-image-bytes")

    def test_utils_convert64_allowed_dirs_containment(self):
        """convert64 enforces containment when allowed_dirs is given,
        resolving symlinks before the check."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            inside = os.path.join(tmp, "inside")
            outside = os.path.join(tmp, "outside")
            os.makedirs(inside)
            os.makedirs(outside)

            ok_path = os.path.join(inside, "ok.png")
            with open(ok_path, "wb") as f:
                f.write(b"ok")
            secret_path = os.path.join(outside, "secret.png")
            with open(secret_path, "wb") as f:
                f.write(b"secret")

            self.assertIsInstance(Utils.convert64(ok_path, allowed_dirs=[inside]), str)

            with self.assertRaises(ValueError):
                Utils.convert64(secret_path, allowed_dirs=[inside])

            # A symlink inside the allowed dir pointing outside must be caught.
            sneaky = os.path.join(inside, "sneaky.png")
            os.symlink(secret_path, sneaky)
            with self.assertRaises(ValueError):
                Utils.convert64(sneaky, allowed_dirs=[inside])

    def test_cleanup_response_keeps_task_reference(self):
        """Regression §1.10: cleanup_response scheduled client.aclose() with
        loop.create_task() but kept no reference to the task; asyncio holds
        only weak references, so the cleanup could be garbage-collected
        before running. The task must be held until it completes."""
        import asyncio
        from maki import utils as utils_module

        class FakeAsyncClient:
            def __init__(self):
                self.closed = False

            async def aclose(self):
                self.closed = True

        async def main():
            client = FakeAsyncClient()
            Utils.cleanup_response(None, client)
            # A strong reference exists while the task is pending.
            self.assertEqual(len(utils_module._CLEANUP_TASKS), 1)
            while utils_module._CLEANUP_TASKS:
                await asyncio.sleep(0)
            self.assertTrue(client.closed)

        asyncio.run(main())

        # Without a running loop the cleanup happens synchronously.
        client = FakeAsyncClient()
        Utils.cleanup_response(None, client)
        self.assertTrue(client.closed)

    def test_agent_initialization_validation(self):
        """Test that Agent validates initialization parameters"""
        maki = MagicMock()

        # Test valid initialization
        agent = Agent("TestAgent", maki, "researcher", "You are a researcher")
        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(agent.role, "researcher")
        self.assertEqual(agent.instructions, "You are a researcher")

        # Test invalid name
        with self.assertRaises(ValueError):
            Agent("", maki, "researcher", "You are a researcher")

        with self.assertRaises(ValueError):
            Agent(None, maki, "researcher", "You are a researcher")

    def test_agent_manager_add_agent_validation(self):
        """Test that AgentManager validates add_agent parameters"""
        maki = MagicMock()
        manager = AgentManager(maki)

        # Test valid addition
        agent = manager.add_agent("TestAgent", "researcher", "You are a researcher")
        self.assertEqual(agent.name, "TestAgent")

        # Test invalid name
        with self.assertRaises(ValueError):
            manager.add_agent("", "researcher", "You are a researcher")

        with self.assertRaises(ValueError):
            manager.add_agent(None, "researcher", "You are a researcher")

        # Test invalid maki_instance
        with self.assertRaises(TypeError):
            manager.add_agent("TestAgent", "researcher", "You are a researcher", "not_a_maki_instance")

if __name__ == '__main__':
    unittest.main()
