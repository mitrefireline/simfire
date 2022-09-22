import os
from importlib.metadata import version

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["SDL_AUDIODRIVER"] = "dsp"

from . import game  # noqa: F401, E402
from .game import image  # noqa: F401, E402
from .game import managers  # noqa: F401, E402
from .game import sprites  # noqa: F401, E402
from .sim import simulation  # noqa: F401, E402
from .utils import config  # noqa: F401, E402
from .utils import decorators  # noqa: F401, E402
from .utils import generate_cfd_wind_layer  # noqa: F401, E402
from .utils import graph  # noqa: F401, E402
from .utils import layers  # noqa: F401, E402
from .utils import log  # noqa: F401, E402
from .utils import terrain  # noqa: F401, E402
from .utils import units  # noqa: F401, E402
from .world import elevation_functions  # noqa: F401, E402
from .world import fuel_array_functions  # noqa: F401, E402
from .world import parameters  # noqa: F401, E402
from .world import presets  # noqa: F401, E402
from .world import rothermel  # noqa: F401, E402
from .world import wind_mechanics  # noqa: F401, E402

__version__ = version("simfire")

del version
