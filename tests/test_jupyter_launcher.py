"""Ayni profil altinda farkli hesaplarla baslatma hatasina karsi kontroller."""

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from launch_jupyter import jupyter_command, runtime_environment


class LauncherTests(unittest.TestCase):
    def test_different_accounts_never_share_runtime_in_the_same_profile(self):
        profile = {"LOCALAPPDATA": str(ROOT / "example-local-data")}
        first = runtime_environment(ROOT, profile, "S-1-5-21-1001")
        second = runtime_environment(ROOT, profile, "S-1-5-21-1002")
        self.assertNotEqual(first["JUPYTER_RUNTIME_DIR"], second["JUPYTER_RUNTIME_DIR"])
        self.assertEqual(first, runtime_environment(ROOT, profile, "S-1-5-21-1001"))

    def test_old_project_runtime_is_overridden_and_project_kernel_is_preserved(self):
        inherited = {"LOCALAPPDATA": str(ROOT / "example-local-data"),
                     "JUPYTER_RUNTIME_DIR": str(ROOT / ".runtime" / "jupyter" / "runtime"),
                     "IPYTHONDIR": str(ROOT / ".runtime" / "ipython"),
                     "JUPYTER_PATH": "previous-kernel-location"}
        env = runtime_environment(ROOT, inherited, "S-1-5-21-1001")
        self.assertNotIn(".runtime", Path(env["JUPYTER_RUNTIME_DIR"]).parts)
        self.assertNotIn(".runtime", Path(env["IPYTHONDIR"]).parts)
        self.assertTrue(env["JUPYTER_PATH"].startswith(str(ROOT / ".venv" / "share" / "jupyter")))
        self.assertTrue(env["JUPYTER_PATH"].endswith(os.pathsep + "previous-kernel-location"))
        self.assertEqual(inherited["JUPYTER_PATH"], "previous-kernel-location")

    def test_launch_remains_local_with_the_original_notebook(self):
        command = jupyter_command(ROOT)
        self.assertIn("--ServerApp.ip=127.0.0.1", command)
        self.assertIn("notebooks/01_Gereksinimler_ve_Veri_Kontrolu.ipynb", command)
        self.assertIn(f"--ServerApp.root_dir={ROOT}", command)
        self.assertEqual(command[0], str(ROOT / ".venv" / "Scripts" / "python.exe"))


if __name__ == "__main__":
    unittest.main()
