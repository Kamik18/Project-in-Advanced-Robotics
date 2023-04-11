"""
gmr
===

Gaussian Mixture Models (GMMs) for clustering and regression in Python.
"""

__version__ = "1.6.2"

try:
    # Idea from sklearn:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages when
    # the dependencies are not available.
    __GMR_SETUP__
except NameError:
    __GMR_SETUP__ = False


if not __GMR_SETUP__:
    from . import gmm, mvn, utils

    __all__ = ["gmm", "mvn", "utils", "sklearn"]

    from .mvn import MVN
    from .gmm import (gmm, kmeansplusplus_initialization,
                      covariance_initialization)

    __all__.extend(["MVN", "GMM", "kmeansplusplus_initialization", "covariance_initialization"])
