import datetime
import re

import numpy as np

from simfire.enums import BurnStatus
from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config

config = Config("configs/historical_config.yml")
sim = FireSimulation(config)

sim.rendering = True

hist_layer = config.historical_layer

current_time = hist_layer.convert_to_datetime(hist_layer.start_time)
end_time = hist_layer.convert_to_datetime(hist_layer.end_time)
hist_layer.perimeter_deltas
i = 0


def parse_duration(duration_str):
    """Converts a string like '2d 1h 06m 21s' to a timedelta object."""
    pattern = r"(?:(\d+)d)?\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?"
    match = re.match(pattern, duration_str)
    if not match:
        raise ValueError("Invalid duration format")

    days, hours, minutes, seconds = (int(x) if x else 0 for x in match.groups())
    return datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


for time in hist_layer.perimeter_deltas:
    mitigation_iterable = []
    duration = parse_duration(time)
    mitigations = hist_layer.make_mitigations(
        current_time,
        current_time + duration,
    )
    locations = np.argwhere(mitigations != 0)
    try:
        mitigation_iterable = np.insert(locations, 2, BurnStatus.FIRELINE.value, axis=1)
    except IndexError:
        mitigation_iterable = []
    sim.update_mitigation(mitigation_iterable)
    sim.run(time)
    current_time += duration

sim.save_gif()
