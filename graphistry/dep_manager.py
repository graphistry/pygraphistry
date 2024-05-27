import importlib
import subprocess

from .constants import GPU_REQ

class DepManager:
    def __init__(self):
        self.pkgs = {}

    def __getattr__(self, pkg:str):
        # self._add_deps(pkg)
        self._proc_import(pkg)
        try:
            return self.pkgs[pkg]
        except KeyError:
            return None

    def _proc_import(self, pkg:str):
        if pkg in GPU_REQ and self._is_gpu_available():
            self._add_deps(pkg)
        elif pkg not in GPU_REQ:
            self._add_deps(pkg)

    def _is_gpu_available(self):
        try:
            output = subprocess.check_output("nvidia-smi", shell=True)
            return len(output) > 0
        except subprocess.CalledProcessError:
            return False

    def _add_deps(self, pkg:str):
        try:
            pkg_val = importlib.import_module(pkg)
            self.pkgs[pkg] = pkg_val
            setattr(self, pkg, pkg_val)
        except:
            pass


deps = DepManager()
