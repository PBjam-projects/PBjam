"""PBjam package-version lookup.

The installed package metadata is used when available. A development fallback
is returned when the source tree is imported without an installed distribution.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pbjam")
except PackageNotFoundError:
    __version__ = "0+unknown"
