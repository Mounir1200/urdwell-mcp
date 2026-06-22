import unittest

import contextmemory
import contextmemory.storage
import urdwell
import urdwell.storage


class RenameCompatibilityTests(unittest.TestCase):
    def test_legacy_namespace_uses_urdwell_implementation(self):
        self.assertEqual(contextmemory.__version__, urdwell.__version__)
        self.assertIs(contextmemory.storage, urdwell.storage)


if __name__ == "__main__":
    unittest.main()
