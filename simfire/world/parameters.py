from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np


@dataclass
class FuelParticle:
    """
    Set default values here since the paper assumes they're constant. These
    could be changed, but for now it's easier to assume they're constant.

    Parameters:
        h: Low heat content (BTU/lb).
        S_T: Total mineral conetent.
        S_e: Effective mineral content.
        p_p: Oven-dry particle density (lb/fg^3)
    """

    # Low Heat Content (BTU/lb)
    h: float = 8000
    # Total Mineral Content
    S_T: float = 0.0555
    # Effective Mineral Content
    S_e: float = 0.01
    # Oven-dry Particle Density (lb/ft^3)
    p_p: float = 32


@dataclass
class Fuel:
    """
    Class that describes the parameters of a fuel type

    Parameters:
        w_0: Oven-dry Fuel Load (lb/ft^2).
        delta: Fuel bed depth (ft).
        M_x: Dead fuel moisture of extinction.
        sigma: Surface-area-to-volume ratio (ft^2/ft^3).
    """

    # Oven-dry Fuel Load (lb/ft^2)
    w_0: float
    # Fuel bed depth (ft)
    delta: float
    # Dead fuel moisture of extinction
    M_x: float
    # Surface-area-to-volume ratio (ft^2/ft^3)
    sigma: float


@dataclass
class Environment:
    """
    These parameters relate to the environment of the tile. For now we'll
    assume these values are constant over a small area.
    The wind speed and direction can be a constant value, nested sequences,
    or numpy arrays. The FireManager will convert the constant values and
    nested sequences to numpy arrays internally.

    Parameters:
        M_f: Fuel moisture (amount of water in fuel/vegetation). 1-3% for SoCal, usually
             never more than 8% for SoCal.
        U: Wind speed at midflame height (ft/min).
        U_dir: Wind direction at midflame height (degrees). 0 is North, 90 is East, 180
               is South, 270 is West.
    """

    # Fuel Moisture (amount of water in fuel/vegetation)
    # 1-3% for SoCal, usually never more than 8% for SoCal
    M_f: float
    # Wind speed at midflame height (ft/min)
    U: Union[float, Sequence[Sequence[float]], np.ndarray]
    # Wind direction at midflame height (degrees)
    # 0 is North, 90 is East, 180 is South, 270 is West
    U_dir: Union[float, Sequence[Sequence[float]], np.ndarray]
