"""
Package
-------
This will be used in CI to make sure that we can import all of the packages/modules in the
installed package. If it fails, there is likely a `src` imported somewhere instead of a
relative import.
"""

import sys
import traceback
from importlib import import_module
from modulefinder import ModuleFinder

import simfire
from simfire import game, sim, utils, world
from simfire.game import managers

# Find all of the submodules
finder = ModuleFinder()
sub_modules = ["simfire." + m for m in finder.find_all_submodules(simfire)]
sub_modules += ["simfire.game." + m for m in finder.find_all_submodules(game)]
sub_modules += [
    "simfire.game.managers." + m for m in finder.find_all_submodules(managers)
]
sub_modules += ["simfire.utils." + m for m in finder.find_all_submodules(utils)]
sub_modules += ["simfire.world." + m for m in finder.find_all_submodules(world)]
sub_modules += ["simfire.sim." + m for m in finder.find_all_submodules(sim)]

# Import all of the found submodules
for module in sub_modules:
    try:
        import_module(module)
    except ModuleNotFoundError:
        traceback.print_exc()
        print(
            '\nERROR: "src" was likely used as an import. Ensure your imports are '
            "relative in all modules and packages."
        )
        sys.exit(1)

print("\nSuccessfully imported modules!")
