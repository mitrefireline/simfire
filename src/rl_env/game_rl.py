import random
from pathlib import Path
from ..utils.config import Config
from .harness import AgentBasedHarness
from .simulation import RothermelSimulation


def main():
    cfg_path = Path('./config.yml')
    cfg = Config(cfg_path)

    actions = ['none', 'fireline']
    observations = ['mitigation', 'w0', 'elevation', 'wind_speed', 'wind_direction']
    simulation = RothermelSimulation(cfg)

    rl_environment = AgentBasedHarness(simulation, actions, observations)
    state = rl_environment.reset()
    done = False
    final_reward = 0
    while not done:
        action = some_action_func()
        state, reward, done, _ = rl_environment.step(action)
        if cfg.render.inline:
            mitigation = rl_environment.harness_conversion(state[1])
            position = state[0]
            simulation.render('inline', mitigation, position)
        final_reward += reward

    mitigation = rl_environment.harness_conversion(state[1])
    if cfg.render.post_agent:
        simulation.render('post agent', mitigation)
    if cfg.render.post_agent_with_fire:
        simulation.render('post agent with fire', mitigation)


def some_action_func():
    '''
    A dummy function to show how the rl side ingests the state
        and returns a dict() of the fire mitigation stategy
    '''
    fire_mitigation = random.choices([0, 1], weights=[0.85, 0.05])
    return fire_mitigation[0]


if __name__ == '__main__':
    main()
