import importlib

DEPS = [
        'cu_cat',
    ]

class DepManager:
    def __init__(self):
        self.pkgs = {}
        self.deps()

    def __getattr__(self, pkg):
        self._add_deps(pkg)
        try:
            return True, "ok", self.pkgs[pkg], self.pkgs[pkg].__version__
        except KeyError:
            return False, str(pkg)+" not installed", None, None

    def _add_deps(self, pkg:str):
        if pkg not in self.pkgs.keys():
            try:
                pkg_val = importlib.import_module(pkg)
                self.pkgs[pkg] = pkg_val
                setattr(self, pkg, pkg_val)
            except:
                setattr(self, pkg, None)

    def deps(self):
        [self._add_deps(dep) for dep in DEPS]
