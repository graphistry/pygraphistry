from importlib import import_module, __import__
from graphistry.util import setup_logger

logger = setup_logger(__name__)

class DepManager:
    """ 
        This class is a helper to manage dependencies
        It allows for dynamic imports and attribute access
        It is used in the Graphistry Python client to manage optional dependencies
    
    : param pkgs: dict, cache to store imported packages, prevent redudant imports
    : returns: None
    
    **Example**
        ::
    
            deps = DepManager()
            has_dbscan = deps.dbscan
            has_umap = deps.umap
    """    
    def __init__(self):
        self.pkgs = {}  # Cache dict to store imported packages, prevent redundant imports

    def __getattr__(self, pkg:str):
        self._add_deps(pkg)  # Import package
        try:
            return self.pkgs[pkg]  # Return package
        except KeyError:
            return None

    def _add_deps(self, pkg:str):
        try:
            # If cudf is being imported, also import cupy since colab installed cudf on CPUs
            if pkg == 'cudf':
                cupy_val = import_module('cupy')
                self.pkgs['cupy'] = cupy_val
                setattr(self, 'cupy', cupy_val)
            pkg_val = import_module(pkg)
            self.pkgs[pkg] = pkg_val  # store in cache dict
            setattr(self, pkg, pkg_val)  # add pkg to deps instance
        except ModuleNotFoundError:
            logger.debug(f"{pkg} not installed")
        except ImportError:
            logger.error(f"{pkg} installed but misconfigured")

    def import_from(self, pkg:str, name:str):
        try:
            module = __import__(pkg, fromlist=[name])  # like _add_deps, but uses __import__ to get top-level pkg/ modules
            self.pkgs[name] = module
        except ModuleNotFoundError:
            logger.debug(f"{pkg} not installed")
        except ImportError:
            logger.error(f"{pkg} installed but misconfigured")
        except Exception as e:
            logger.warn("Unexpected exn during lazy import", exc_info=e)


deps = DepManager()
