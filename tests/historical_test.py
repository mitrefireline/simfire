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
    config.operational.height,
    config.operational.width,
)

# Get hist_layer start time
# Set udpate duration to 1m, 1h, whatever makes sense
# Loop through simulation
# Update Historical Layer to get mitigations for specifc times (can filter the data frame)

print("done!")
