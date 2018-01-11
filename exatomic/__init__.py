# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
A unified data anlaysis and visualization platform for computational and
theoretical chemists, physicists, etc. Support for molecular geometry and
orbital visualization is provided via the `Jupyter`_ notebook, a web-browser
based interactive (multi-programming language) environment.

.. extended description (todo)

Warning:
    This package uses the `atomic`_ unit system (Hartree) by default.

.. _atomic: https://en.wikipedia.org/wiki/Atomic_units

Supported Software
---------------------
The list below contains third-party software that is supported by this package.
For specific features supported (per software), see the appropriate description
below.

- :mod:`~exatomic.adf.__init__`: `Amsterdam Density Functional`_
- :mod:`~exatomic.gaussian.__init__`: `Gaussian`_
- :mod:`~exatomic.molcas.__init__`: `OpenMolcas`_
- :mod:`~exatomic.nbo.__init__`: `NBO`_
- :mod:`~exatomic.nwchem.__init__`: `NWChem`_
- :mod:`~exatomic.qe.__init__`: `Quantum ESPRESSO`_
- :mod:`~exatomic.interfaces.__init__`: Additional 3rd party support

.. _Jupyter: https://jupyter.org
.. _Amsterdam Density Functional: https://www.scm.com
.. _Gaussian: http://gaussian.com/
.. _OpenMolcas: https://gitlab.com/Molcas/OpenMolcas
.. _NBO: http://nbo6.chem.wisc.edu/
.. _NWChem: http://www.nwchem-sw.org/index.php/Main_Page
.. _Quantum ESPRESSO: http://www.quantum-espresso.org/
"""
def _jupyter_nbextension_paths():
    """Jupyter notebook extension directory paths."""
    return [{
        'section': "notebook",
        'src': "static/js",
        'dest': "exatomic",
        'require': "exatomic/extension"
    }]


__js_version__ = "^0.4.6"
from ._version import __version__
from . import core
from .core import Universe, Editor, Atom, AtomicField, Frame
from .interfaces import XYZ, Cube
from .widgets import DemoContainer, DemoUniverse, UniverseWidget, TensorContainer
# from . import widgets
