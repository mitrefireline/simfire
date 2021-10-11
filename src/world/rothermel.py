from math import atan2, cos, exp, radians
from typing import Tuple

from src.world.parameters import Environment, FuelArray, FuelParticle


def compute_rate_of_spread(loc: Tuple[float, float, float], new_loc: Tuple[float, float,
                                                                           float],
                           fuel_arr: FuelArray, fuel_particle: FuelParticle,
                           environment: Environment) -> float:
    # Mineral Damping Coefficient
    eta_S = min(0.174 * fuel_particle.S_e**-0.19, 1)
    # Moisture Damping Coefficient
    r_M = min(environment.M_f / fuel_arr.fuel.M_x, 1)
    eta_M = 1 - 2.59 * r_M + 5.11 * r_M**2 - 3.52 * r_M**3
    # Net Fuel Load (lb/ft^2)
    w_n = fuel_arr.fuel.w_0 * (1 - fuel_particle.S_T)
    # Oven-dry Bulk Density (lb/ft^3)
    p_b = fuel_arr.fuel.w_0 / fuel_arr.fuel.delta
    # Packing Ratio
    B = p_b / fuel_particle.p_p
    # Optimum Packing Ratio
    B_op = 3.348 * fuel_arr.fuel.sigma**-0.8189
    # Maximum Reaction Velocity (1/min)
    gamma_prime_max = fuel_arr.fuel.sigma**1.5 / (495 + 0.0594 * fuel_arr.fuel.sigma**1.5)
    A = 133 * fuel_arr.fuel.sigma**-0.7913
    # Optimum Reaction Velocity (1/min)
    gamma_prime = gamma_prime_max * (B / B_op)**A * exp(A * (1 - B / B_op))
    # Reaction Intensity (BTU/ft^2-min)
    I_R = gamma_prime * w_n * fuel_particle.h * eta_M * eta_S
    # Propagating Flux Ratio
    xi = exp((0.792 + 0.681 * fuel_arr.fuel.sigma**0.5) *
             (B + 0.1)) / (192 + 0.25 * fuel_arr.fuel.sigma)

    # Wind Factor
    c = 7.47 * exp(-0.133 * fuel_arr.fuel.sigma**0.55)
    b = 0.02526 * fuel_arr.fuel.sigma**0.54
    e = 0.715 * exp(-3.59e-4 * fuel_arr.fuel.sigma)
    # Need to find wind component in direction of travel
    # Switch order of y-component subtraction since image y coordintates
    # increase from top to bottom
    angle_of_travel = atan2(loc[1] - new_loc[1], new_loc[0] - loc[0])
    # Subtract 90 degrees because this angle is North-oriented
    wind_angle_radians = radians(90 - environment.U_dir)
    wind_along_angle_of_travel = environment.U * \
                                 cos(wind_angle_radians - angle_of_travel)
    # This is the wind speed in in this direction
    U = wind_along_angle_of_travel
    # Negative wind leads to trouble with calculation and doesn't
    # physically make sense
    # theta_t_d = math.degrees(angle_of_travel)
    # theta_w_d = math.degrees(wind_angle_radians)
    U = max(U, 0)
    phi_w = c * U**b * (B / B_op)**-e

    # Slope Factor
    # Phi is the slope between the two locations (i.e. the change in elevation).
    # We can approximate this using the z coordinates for each point
    # This equation normally has tan(phi)**2, but we can substitute
    # tan(phi) = (new_loc.z-old_loc.z)
    phi_s = 5.275 * B**-0.3 * (new_loc[2] - loc[2])**2

    # Effective Heating Number
    epsilon = exp(-138 / fuel_arr.fuel.sigma)
    # Heat of Preignition (BTU/lb)
    Q_ig = 250 + 1116 * environment.M_f

    # Rate of Spread (ft/min)
    R = (I_R * xi * (1 + phi_w + phi_s)) / (p_b * epsilon * Q_ig)

    # Take the minimum with 0 because a fire cannot put itself out
    R = max(R, 0)

    return R
