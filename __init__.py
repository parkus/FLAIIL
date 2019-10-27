"""
FLAIIL: FLAre Identification in Interrupted Lightcurves
"""
from __future__ import division, print_function, absolute_import

from . import numbers
from . import ranges
from . import plots
from . import identify
from .identify import identify_flares
from .qmodel import QuiescenceModel
from .plots import standard_flareplot
