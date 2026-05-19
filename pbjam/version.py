from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pbjam")
except PackageNotFoundError:
    __version__ = "0+unknown"
