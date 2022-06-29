"""
This module contains preset FuelArray tile values for various terrain/fuel.
Four significant figures were used for rounding
For fuel load (w_0), the 1-hour fuel load is used
For surface-area-to-volume (SAV) ratio (sigma), the characteristic SAV is used
For M-x, the Dead fuel moisture of extinction is used
Fir delta, the Fuel bed depth is used

As specified in: https://www.fs.fed.us/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf

Urban, Snow/Ice, Agricutlture, Water, Barren are Non-Burnable fuel types as described in:
https://gacc.nifc.gov/oncc/docs/40-Standard%20Fire%20Behavior%20Fuel%20Models.pdf
"""
from .parameters import Fuel

ShortGrass = Fuel(w_0=0.0340, delta=1.000, M_x=0.1200, sigma=3500)

GrassTimberShrubOverstory = Fuel(w_0=0.0918, delta=1.000, M_x=0.1500, sigma=2784)

TallGrass = Fuel(w_0=0.1377, delta=2.500, M_x=0.2500, sigma=1500)

Chaparral = Fuel(w_0=0.2296, delta=6.000, M_x=0.2000, sigma=1739)

Brush = Fuel(w_0=0.0459, delta=2.000, M_x=0.2000, sigma=1683)

DormantBrushHardwoodSlash = Fuel(w_0=0.0688, delta=2.500, M_x=0.25, sigma=1564)

SouthernRough = Fuel(w_0=0.0459, delta=2.500, M_x=0.4000, sigma=1552)

ClosedShortNeedleTimberLitter = Fuel(w_0=0.0688, delta=0.2000, M_x=0.3000, sigma=1889)

HardwoodLongNeedlePineTimber = Fuel(w_0=0.1331, delta=0.2000, M_x=0.2500, sigma=2484)

TimberLitterUnderstory = Fuel(w_0=0.1377, delta=1.000, M_x=0.2500, sigma=1764)

LightLoggingSlash = Fuel(w_0=0.0688, delta=1.000, M_x=0.1500, sigma=1182)

MediumLoggingSlash = Fuel(w_0=0.1836, delta=2.300, M_x=0.2000, sigma=1145)

HeavyLoggingSlash = Fuel(w_0=0.3214, delta=3.000, M_x=0.2500, sigma=1159)

ShortSparseDryClimateGrass = Fuel(w_0=0.0046, delta=0.4000, M_x=0.1500, sigma=2054)

NBUrban = Fuel(w_0=0.0, delta=1.000, M_x=1.000, sigma=1.000)

NBSnowIce = Fuel(w_0=0.0, delta=1.000, M_x=1.000, sigma=1.000)

NBWater = Fuel(w_0=0.0, delta=1.000, M_x=1.000, sigma=1.000)

NBAgriculture = Fuel(w_0=0.0, delta=1.000, M_x=1.000, sigma=1.000)

NBBarren = Fuel(w_0=0.0, delta=1.000, M_x=1.000, sigma=1.000)

NBNoData = Fuel(w_0=0.0, delta=1.000, M_x=1.000, sigma=1.000)
