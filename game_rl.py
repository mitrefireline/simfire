import pygame
import random
from pathlib import Path
from src.utils.config import Config
from src.rl_env.fireline_env import FireLineEnv, RLEnv
from src.rl_env.harness import RLEnvironment
from src.rl_env.simulation import RothermalSimulation


def main():
    cfg_path = Path('./config.yml')
    cfg = Config(cfg_path)
    seed = None
    if seed is not None:
        env_config = FireLineEnv(cfg, seed)
    else:
        env_config = FireLineEnv(cfg)
    rl_environment = RLEnv(env_config)
    state = rl_environment.reset()
    done = False
    final_reward = 0
    while not done:
        action = some_action_func(state)
        state, reward, done, _ = rl_environment.step(action)
        final_reward += reward

    pygame.quit()


def main_v2():
    cfg_path = Path('./config.yml')
    cfg = Config(cfg_path)

    actions = ['none', 'fireline']
    observations = ['mitigation', 'w0', 'elevation', 'wind_speed', 'wind_direction']
    simulation = RothermalSimulation(cfg)

    rl_environment = RLEnvironment(simulation, actions, observations)
    state = rl_environment.reset()
    done = False
    final_reward = 0
    while not done:
        action = some_action_func(state)
        state, reward, done, _ = rl_environment.step(action)
        final_reward += reward


def some_action_func(state):
    '''
    A dummy function to show how the rl side ingests the state
        and returns a dict() of the fire mitigation stategy
    '''
    fire_mitigation = random.choices([0, 1], weights=[0.85, 0.05])
    return fire_mitigation[0]


if __name__ == '__main__':
    main_v2()
