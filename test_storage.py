import os
import tempfile
import unittest
from pathlib import Path

from contextmemory import storage
from contextmemory.storage import JsonStore


class DefaultDataDirTests(unittest.TestCase):
    def test_default_is_outside_the_installed_package(self):
        package_dir = Path(storage.__file__).resolve().parent
        default = storage.default_data_dir().resolve()
        self.assertFalse(
            str(default).startswith(str(package_dir)),
            "memories must not live inside the package; an upgrade would erase them",
        )

    def test_explicit_data_dir_argument_wins(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JsonStore(Path(temp_dir) / "explicit")
            self.assertEqual(store.dir, Path(temp_dir) / "explicit")

    def test_environment_variable_overrides_default(self):
        previous = os.environ.get(storage.DATA_DIR_ENV_VAR)
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ[storage.DATA_DIR_ENV_VAR] = temp_dir
            try:
                store = JsonStore()
                self.assertEqual(store.dir, Path(temp_dir))
            finally:
                if previous is None:
                    os.environ.pop(storage.DATA_DIR_ENV_VAR, None)
                else:
                    os.environ[storage.DATA_DIR_ENV_VAR] = previous


if __name__ == "__main__":
    unittest.main()
