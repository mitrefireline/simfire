import datetime
import re

from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config

config = Config("configs/historical_config.yml")
sim = FireSimulation(config)

sim.rendering = True

hist_layer = config.historical_layer

current_time = hist_layer.convert_to_datetime(hist_layer.start_time)
current_time += datetime.timedelta(days=2, hours=13)
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


# for time in hist_layer.perimeter_deltas:
#     duration = parse_duration(time)
duration = "1h"
datetime_duration = parse_duration(duration)
while current_time < end_time:
    mitigation_points = hist_layer.get_mitigations_by_time(
        current_time,
        current_time + datetime_duration,
    )
    sim.update_mitigation(mitigation_points)
    sim.run(duration)
    current_time += datetime_duration

sim.save_gif()
