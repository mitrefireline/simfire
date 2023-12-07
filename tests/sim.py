from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config

config = Config("configs/functional_config.yml")
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
