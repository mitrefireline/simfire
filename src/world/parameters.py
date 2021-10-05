from dataclasses import dataclass, field


@dataclass
class FuelParticle:
    '''
    Set default values here since the paper assumes they're constant. These
    could be changed, but for now it's easier to assume they're constant.
    '''
    # Low Heat Content (BTU/lb)
    h: float = 8000
    # Total Mineral Content
    S_T: float = 0.0555
    # Effective Mineral Content
    S_e: float = 0.01
    # Oven-dry Particle Density (lb/ft^3)
    p_p: float = 32


@dataclass
class Tile:
    # Tile length in x-direction (ft)
    x: float
    # Tile length in y-direction (ft)
    y: float
    # Tile elevation (ft)
    z: float
    # Area of the tile (ft^2)
    area: float = field(init=False)
    
    def __post_init__(self):
        self.area = self.x * self.y


@dataclass
class FuelArray:
    '''
    These parameters relate to the fuel in a tile. Need a Tile as a
    parameter to get area and volume information.
    '''
    # Tile on which the fuel exists
    tile: Tile
    # Oven-dry Fuel Load (lb/ft^2)
    w_0: float
    # Fuel bed depth (ft)
    delta: float
    # Dead fuel moisture of extinction
    M_x: float
    # Surface-area-to-volume ratio (ft^2/ft^3)
    sigma: float = field(init=False)
    
    def __post_init__(self):
        # Can simplify to 1/delta since the assumption is each tile has
        # constant fuel array height
        self.sigma = 1 / self.delta


@dataclass
class Environment:
    '''
    These parameters relate to the environment of the tile. For now we'll
    assume these values are constant over a small area.
    '''
    # Moisture Content
    M_f: float
    # Wind Velocity at midflame height (ft/min)
    U: float
