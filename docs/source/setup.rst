Installation
============

PBjam requires Python 3.10 or newer. We recommend installing it in a virtual
environment.

Install the latest release from PyPI with::

   python -m pip install pbjam

To install the documentation or test dependencies as well, use::

   python -m pip install "pbjam[docs]"
   python -m pip install "pbjam[test]"

For development, clone the repository and install it in editable mode::

   git clone https://github.com/PBjam-projects/PBjam.git
   cd PBjam
   python -m pip install -e ".[test,docs]"
