import os
from importlib.metadata import version

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

__version__ = version("simfire")

del version
