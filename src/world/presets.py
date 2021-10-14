import numpy as np
'''
This module contains preset FuelArray tile values for various terrain/fuel.
Four significant figures were used for rounding
For fuel load (w_0), the 1-hour fuel load is used
For surface-area-to-volume (SAV) ratio, the characteristic SAV is used
'''
from .parameters import Fuel

ShortGrass = Fuel(w_0=0.0340, delta=1.000, M_x=0.1200, sigma=3500)

TallGrass = Fuel(w_0=0.1377, delta=2.500, M_x=0.2500, sigma=1500)

Chaparral = Fuel(w_0=0.2296, delta=6.000, M_x=0.2000, sigma=1739)


Brush = Fuel(w_0=0.0459, delta=2.000, M_x=0.2000, sigma=1683)

ShortSparseDryClimateGrass = Fuel(w_0=0.0046, delta=0.4000, M_x=0.1500, sigma=2054)
