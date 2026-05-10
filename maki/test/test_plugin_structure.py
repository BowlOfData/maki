"""
Repository-level verification for built-in plugin package consistency.
"""

from pathlib import Path

from maki.config import PLUGIN_REQUIRED_FILES
from maki.plugins import PLUGIN_REGISTRY, get_plugin_class, list_plugins


def test_plugin_registry_matches_package_layout():
    plugin_root = Path(__file__).resolve().parent.parent / "plugins"

    assert sorted(PLUGIN_REGISTRY) == list_plugins()

    for plugin_name, class_name in PLUGIN_REGISTRY.items():
        plugin_dir = plugin_root / plugin_name
        assert plugin_dir.is_dir(), f"Missing plugin directory: {plugin_name}"

        for required_file in PLUGIN_REQUIRED_FILES:
            assert (plugin_dir / required_file).is_file(), (
                f"Plugin '{plugin_name}' is missing required file '{required_file}'"
            )

        assert (plugin_dir / f"{plugin_name}.py").is_file(), (
            f"Plugin '{plugin_name}' is missing its implementation module"
        )
        assert (plugin_dir / f"test_{plugin_name}.py").is_file(), (
            f"Plugin '{plugin_name}' is missing its plugin-specific test module"
        )

        plugin_class = get_plugin_class(plugin_name)
        assert plugin_class.__name__ == class_name
