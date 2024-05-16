# SimFire Fire Simulator

<p align="center">
    <img src="https://raw.githubusercontent.com/mitrefireline/simfire/main/assets/icons/rl_logo_horizontal.png">
</p>

## Introduction

SimFire uses [PyGame](https://www.pygame.org/wiki/about) to display and simulate different fire spread models, including the Rothermel Surface fire spread model described in [this](https://www.fs.usda.gov/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf) paper.

For more comprehensive documentation, go to our [docs page](https://mitrefireline.github.io/simfire).

This repository is part of the MITRE Fireline project and is associated with both [BurnMD](https://github.com/mitrefireline/burnmd) and [SimHarness](https://github.com/mitrefireline/simharness). BurnMD is used in the `HistoricalDataLayer` to provide historical fire data for the simulation. SimHarness is a reinforcement learning training harness that uses this simulator to train agents to fight fires.

## Running the Simulation
<figure>
    <p align="center">
        <img src="https://raw.githubusercontent.com/mitrefireline/simfire/main/assets/gifs/simulation_33.06N_116.58W.gif" width="225" />
        <img src="https://raw.githubusercontent.com/mitrefireline/simfire/main/assets/gifs/simulation_39.67N_119.80W.gif" width="225" />
        <br><b>Left: Fire simulated near Julian, CA. Right: Fire simulated near Reno, NV.
        <br>Both fires have winds from the east at 20mph<b>
    </p>
 </figure>

Install `simfire` by following the [installation instructions](#installing-the-package). Then run the `run_game.py` script:

```shell
python run_game.py
```

### Running as a Python Module

```python
from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config

config = Config("configs/operational_config.yml")
sim = FireSimulation(config)

# Run a 1 hour simulation
sim.run("1h")

# Run the same simulation for 30 more minutes
sim.run("30m")

# Render the next 2 hours of simulation
sim.rendering = True
sim.run("2h")

# Now save a GIF and fire spread graph from the last 2 hours of simulation
sim.save_gif()
sim.save_spread_graph()
# Saved to the location specified in the config: simulation.sf_home

# Update agents for display
# (x, y, agent_id)
agent_0 = (5, 5, 0)
agent_1 = (5, 5, 1)

agents = [agent_0, agent_1]

# Create the agents on the display
sim.update_agent_positions(agents)

# Loop through to move agents
for i in range(5):
    agent_0 = (5 + i, 5 + i, 0)
    agent_1 = (5 + i, 5 + i, 1)
    # Update the agent positions on the simulation
    sim.update_agent_positions([agent_0, agent_1])
    # Run for 1 update step
    sim.run(1)

# Turn off rendering so the display disappears and the simulation continues to run in the
# background
sim.rendering = False
```

## Installing the Package

```shell
pip install simfire
```

## Contributing

For contributing, see the [Contribution Page](https://mitrefireline.github.io/simfire/contributing.html) in our [docs](https://mitrefireline.github.io/simfire).

## Citation

To cite this software, use the “Cite this repository” link built into GitHub on the right.

### Copyright
Copyright ©2023 The MITRE Corporation. ALL RIGHTS RESERVED. Approved for Public Release; Distribution Unlimited. Public Release Case Number 22-3261.
