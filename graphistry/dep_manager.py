import importlib

class DepManager:
    def __init__(self):
        self.pkgs = {}

    def __getattr__(self, pkg:str):
        self._add_deps(pkg)
        try:
            return True, "ok", self.pkgs[pkg], self.pkgs[pkg].__version__
        except KeyError:
            return False, str(pkg) + " not installed", None, None

    def _add_deps(self, pkg:str):
        try:
            pkg_val = importlib.import_module(pkg)
            self.pkgs[pkg] = pkg_val
            # setattr(self, pkg, pkg_val)
        except:
            setattr(self, pkg, None)
