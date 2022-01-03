from typing import Dict, List
import numpy as np


def SimulationConversion(sim_attributes: Dict[str, int],
                         harness_attributes: List[str]) -> np.ndarray:
    '''
    This function will convert the returns of the Simulation.get_attributes()
        to the RL harness np.ndarray structure

    Attributes:
        sim_attributes: Dict[str, np.ndarray]
            A Dict of attrbutes and the associated arrays

        harness_attributes: List[str]
            A list of strings of the desired attributes to include in the RL harness

    Returns:
        np.ndarray
            A numpy array of the converted attributes for the RL harness to use

    '''
    res = []
    for harness_attr in harness_attributes:
        if harness_attr in sim_attributes.keys():
            res.append(sim_attributes[harness_attr])

    res = np.asarray(res)

    return res


def ActionsToInt(harness_actions: List[str]) -> List[int]:
    '''
    This function will convert the actions to a list of integers that is
        zero-indexed

    Attributes:
        List[str]
            A list of the actions the RL harness will use

    Returns:
        List[int]
            A zero-indexed list

    '''

    return [int(x) for x in range(len(harness_actions))]


def HarnessConversion(mitigation_map: np.ndarray, sim_actions: Dict[str, int],
                      harness_actions: List[str]) -> np.ndarray:
    '''
    This function will convert the returns of the Simulation.get_actions()
        to the RL harness List of ints structure where the simulation action
        integer starts at index 0 for the RL harness

    Example:    mitigation_map = (0, 1, 1, 0)
                sim_action = {'none': 0, 'fireline':1, 'scratchline':2, 'wetline':3}
                harness_actions = ['none', 'scratchline']

            Harness                  Simulation
            ---------               ------------
            'none': 0           -->   'none': 0
            'scratchline': 1    -->   'scratchline': 2

            return (0, 2, 2, 0)

    Attributes:
        mitigation_map: np.ndarray
            A np.ndarray of the harness mitigation map

        sim_actions: Dict[str, int]
            A Dict of attrbutes and the associated ints

        harness_attributes: List[str]
            A list of strings of the desired actions to include in the RL harness

    Returns:
        np.ndarray
            A np.ndarray of the converted mitigation map from RL harness to the correct
                Simulation BurnStatus types

    '''

    harness_ints = np.unique(mitigation_map)
    harness_dict = {
        harness_actions[i]: harness_ints[i]
        for i in range(len(harness_actions))
    }

    sim_mitigation_map = []
    for mitigation_i in mitigation_map:
        for mitigation_j in mitigation_i:
            action = [key for key, value in harness_dict.items() if value == mitigation_j]
            sim_mitigation_map.append(sim_actions[action[0]])

    return np.asarray(sim_mitigation_map).reshape(len(mitigation_map[0]),
                                                  len(mitigation_map[1]))
