import pygame
import random
import src.config as cfg
from src.rl_env.fireline_env import FireLineEnv, RLEnv


def main():
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


def some_action_func(state):
    '''
    A dummy function to show how the rl side ingests the state
        and returns a dict() of the fire mitigation stategy
    '''
    fire_mitigation = random.choices([0, 1], weights=[0.95, 0.05])
    return fire_mitigation[0]


if __name__ == '__main__':
    main()
