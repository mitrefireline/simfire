from datetime import timedelta

import numpy as np

from simfire.enums import BurnStatus
from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config

config = Config("configs/historical_config.yml")
sim = FireSimulation(config)

sim.rendering = True

hist_layer = config.historical_layer

update_minutes = 1 * 60
update_interval = f"{update_minutes}m"
update_interval_datetime = timedelta(minutes=update_minutes)
current_time = hist_layer.convert_to_datetime(hist_layer.start_time)
end_time = hist_layer.convert_to_datetime(hist_layer.end_time)
while current_time < end_time:
    mitigation_iterable = []
    mitigations = hist_layer.make_mitigations(
        current_time, current_time + update_interval_datetime
    )
    locations = np.argwhere(mitigations != 0)
    try:
        mitigation_iterable = np.insert(locations, 2, BurnStatus.FIRELINE.value, axis=1)
    except IndexError:
        mitigation_iterable = []
    sim.update_mitigation(mitigation_iterable)
    sim.run(update_interval)
    current_time += update_interval_datetime

sim.save_gif()
