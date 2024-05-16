from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config
from simfire.utils.layers import HistoricalLayer

config = Config("configs/historical_config.yml")
sim = FireSimulation(config)

hist_layer = HistoricalLayer(
    config.historical.year,
    config.historical.state,
    config.historical.fire,
    config.historical.path,
    config.area.screen_size,
)

print("done!")
