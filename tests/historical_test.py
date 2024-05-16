from datetime import datetime, timedelta
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
update_minutes = 12 * 60
update_interval = f"{update_minutes}m"
update_interval_datetime = timedelta(minutes=update_minutes)
current_time = hist_layer.convert_to_datetime(hist_layer.start_time)
end_time = hist_layer.convert_to_datetime(hist_layer.end_time)
while current_time < end_time:
    mitigations = hist_layer.make_mitigations(
        current_time, current_time + update_interval_datetime
    )
    sim.fire_map[mitigations != 0] = mitigations[mitigations != 0]
    sim.run(update_interval)
    current_time += update_interval_datetime
# Loop through simulation
# Update Historical Layer to get mitigations for specifc times (can filter the data frame)

print("done!")
