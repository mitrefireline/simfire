from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config
from simfire.utils.layers import HistoricalLayer

config = Config("configs/historical_config.yml")
sim = FireSimulation(config)

hist_layer = HistoricalLayer(2018, "Arizona", "Rattlesnake", "../burnmd/")
