import unittest

import numpy as np

from ..parameters import FuelParticle
from ..presets import Chaparral, TallGrass
from ..rothermel import compute_rate_of_spread
from ... import config as cfg

KNOWN_ROTHERMEL_OUTPUT = [
    16.672710418701172, 16.672710418701172, 44.74728775024414, 62.592369079589844,
    20.221668243408203, 6.884253025054932, 6.884253025054932, 6.884253025054932
]


class TestRothermel(unittest.TestCase):
    def test_compute_rate_of_spread(self) -> None:
        chaparral = Chaparral
        grass = TallGrass
        particle = FuelParticle()

        loc_x = [1] * 8
        loc_y = [1] * 8
        loc_z = [0] * 8

        new_loc_x = [1, 2, 2, 2, 1, 0, 0, 0]
        new_loc_y = [2, 2, 1, 0, 0, 0, 1, 2]
        new_loc_z = [0] * 8

        w_0 = [chaparral.w_0] * 4 + [grass.w_0] * 4
        delta = [chaparral.delta] * 4 + [grass.delta] * 4
        M_x = [chaparral.M_x] * 4 + [grass.M_x] * 4
        sigma = [chaparral.sigma] * 4 + [grass.sigma] * 4

        h = [particle.h] * 8
        S_T = [particle.S_T] * 8
        S_e = [particle.S_e] * 8
        p_p = [particle.p_p] * 8

        M_f = [cfg.M_f] * 8
        U = [cfg.U] * 8
        U_dir = [cfg.U_dir] * 8

        loc_x = np.array(loc_x, dtype=np.float32)
        loc_y = np.array(loc_y, dtype=np.float32)
        loc_z = np.array(loc_z, dtype=np.float32)

        new_loc_x = np.array(new_loc_x, dtype=np.float32)
        new_loc_y = np.array(new_loc_y, dtype=np.float32)
        new_loc_z = np.array(new_loc_z, dtype=np.float32)

        w_0 = np.array(w_0, dtype=np.float32)
        delta = np.array(delta, dtype=np.float32)
        M_x = np.array(M_x, dtype=np.float32)
        sigma = np.array(sigma, dtype=np.float32)

        h = np.array(h, dtype=np.float32)
        S_T = np.array(S_T, dtype=np.float32)
        S_e = np.array(S_e, dtype=np.float32)
        p_p = np.array(p_p, dtype=np.float32)

        M_f = np.array(M_f, dtype=np.float32)
        U = np.array(U, dtype=np.float32)
        U_dir = np.array(U_dir, dtype=np.float32)

        R = compute_rate_of_spread(loc_x, loc_y, loc_z, new_loc_x, new_loc_y, new_loc_z,
                                   w_0, delta, M_x, sigma, h, S_T, S_e, p_p, M_f, U,
                                   U_dir)
        self.assertListEqual(R.tolist(), KNOWN_ROTHERMEL_OUTPUT)
