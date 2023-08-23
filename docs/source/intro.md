# Introduction

SimFire uses [PyGame](https://www.pygame.org/wiki/about) to display and simulate different fire spread models, including the Rothermel Surface fire spread model described in [this](https://www.fs.usda.gov/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf) paper.

[GitLab Page](https://github.com/mitrefireline/simfire)

## Running the Simulation

### Installing via Pip

```shell
pip install simfire
```

### Installing from Source

Clone the repository:

```shell
git clone git@github.com:mitrefireline/simfire.git
```

**NOTE**: Make sure you're using Python 3.9, due to PyGame (scheduled to be removed).

Then, install the requirements:

```shell
export POETRY_VERSION=1.4.0
curl -sSkL https://install.python-poetry.org | python -
export PATH=$PATH:$HOME/.local/bin
poetry install --no-dev
```

And run the `run_game.py` script:

```shell
python run_game.py
```

### Running as a Python Module

```python
from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation

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
# Saved to the location specified in the config: simulation.save_path

# Update agents for display
# (x, y, agent_id)
agent_0 = (5, 78, 0)
agent_1 = (80, 105, 1)

agents = [agent_0, agent_1]

# Create the agents on the display
sim.update_agent_positions(agents)

# Loop through to move agents
for i in range(60):
    # Do something here to choose the new agent locations
    agent_0 = (new_col, new_row, 0)
    agent_1 = (new_col, new_row, 1)
    # Update the agent positions on the simulation
    sim.update_agent_positions([agent_0, agent_1])
    # Run for 1 update step
    sim.run(1)

# Turn off rendering so the display disappears and the simulation continues to run in the background
sim.rendering = False
```

## Setting Up xpra for Remote Simulation Visualization

If you'd like, you can modify your `.bashrc` (or `.zshrc`, depending on your terminal) file to easily forward the simulation display from your remote machine.

Before adding to the `.bashrc`, install the following packages on your remote terminal, which will open a persistent tmux session titled `xpra` to forward your remote ports:

```shell
sudo apt install xpra
sudo apt install tmux
```

Add the following to your `.bashrc`, making sure to substitute your own `<DISPLAY>` from the section above:

```shell

# Added for XPRA support
if ! tmux has-session -t xpra
then
    tmux new -s xpra -d
    tmux send -t xpra.0 'xpra start :<DISPLAY> --systemd-run=no --daemon=no' Enter
fi

```

Finally, on your local terminal, install the `xpra` package and then run the following command:

```shell
xpra attach --ssh=ssh ssh://<CONTAINER>@rlord/<DISPLAY>
```
