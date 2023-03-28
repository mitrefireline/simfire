import numpy as np


def compute_rate_of_spread(
    loc_x: np.ndarray,
    loc_y: np.ndarray,
    new_loc_x: np.ndarray,
    new_loc_y: np.ndarray,
    w_0: np.ndarray,
    delta: np.ndarray,
    M_x: np.ndarray,
    sigma: np.ndarray,
    h: np.ndarray,
    S_T: np.ndarray,
    S_e: np.ndarray,
    p_p: np.ndarray,
    M_f: np.ndarray,
    U: np.ndarray,
    U_dir: np.ndarray,
    slope_mag: np.ndarray,
    slope_dir: np.ndarray,
) -> np.ndarray:
    """
    Compute the basic Rothermel rate of spread. All measurements are assumed to be in
    feet, minutes, and pounds, and BTU.

    Arguments:
        loc_x: The current x location
        loc_y: The current y location
        loc_z: The current z elevation
        loc_x: The new x location
        loc_y: The new y location
        loc_z: The new z elevation
        w_0: The oven-dry fuel load of the fuel at the new location
        delta: The fuel bed depth of the fuel at the new location
        M_x: The dead fuel moisture of extinction of the fuel at the new location
        sigma: The Surface-area-to-volume ratio of the fuel at the new location
        h: The fuel particle low heat content
        S_T: The fuel particle total mineral content
        S_e: The fuel particle effective mineral content
        p_p: The fuel particle oven-dry particle density
        M_f: The envrionment fuel moisture
        U: The envrionment wind speed
        U_dir: The envrionment wind direction (degrees clockwise from North)
        slope_dir: The angle of the steepest ascent at the location

    Returns:
        R: The computed rate of spread in ft/min
    """
    # Check for non-burnable fuel and return 0 (no spread)
    if w_0.all() == 0:
        return np.zeros_like(loc_x)
    # Mineral Damping Coefficient
    eta_S = np.minimum(0.174 * S_e**-0.19, np.ones_like(S_e))
    # Moisture Damping Coefficient
    r_M = np.minimum(M_f / M_x, np.ones_like(M_f))
    eta_M = 1 - 2.59 * r_M + 5.11 * r_M**2 - 3.52 * r_M**3
    # Net Fuel Load (lb/ft^2)
    w_n = w_0 * (1 - S_T)
    # Oven-dry Bulk Density (lb/ft^3)
    p_b = w_0 / delta
    # Packing Ratio
    B = p_b / p_p
    # Optimum Packing Ratio
    B_op = 3.348 * sigma**-0.8189
    # Maximum Reaction Velocity (1/min)
    gamma_prime_max = sigma**1.5 / (495 + 0.0594 * sigma**1.5)
    A = 133 * sigma**-0.7913
    # Optimum Reaction Velocity (1/min)
    gamma_prime = gamma_prime_max * (B / B_op) ** A * np.exp(A * (1 - B / B_op))
    # Reaction Intensity (BTU/ft^2-min)
    I_R = gamma_prime * w_n * h * eta_M * eta_S
    # Propagating Flux Ratio
    xi = np.exp((0.792 + 0.681 * sigma**0.5) * (B + 0.1)) / (192 + 0.25 * sigma)

    # Wind Factor
    c = 7.47 * np.exp(-0.133 * sigma**0.55)
    b = 0.02526 * sigma**0.54
    e = 0.715 * np.exp(-3.59e-4 * sigma)
    # Need to find wind component in direction of travel
    # Switch order of y-component subtraction since image y coordintates
    # increase from top to bottom
    angle_of_travel = np.arctan2(loc_y - new_loc_y, new_loc_x - loc_x)
    # Subtract 90 degrees because this angle is North-oriented
    wind_angle_radians = np.radians(90 - U_dir)
    wind_along_angle_of_travel = U * np.cos(wind_angle_radians - angle_of_travel)
    # This is the wind speed in in this direction
    U = wind_along_angle_of_travel
    # Negative wind leads to trouble with calculation and doesn't
    # physically make sense
    U = np.maximum(U, np.zeros_like(U))
    phi_w = c * U**b * (B / B_op) ** -e

    # Slope Factor
    # Phi is the slope between the two locations (i.e. the change in elevation).
    # The model calls for the tangent of the slope angle between the two points,
    # but we can approximate this by projecting the slope along the direction
    # of travel
    slope_along_angle_of_travel = -slope_mag * np.cos(slope_dir + angle_of_travel)
    sign = -1 + 2 * (slope_along_angle_of_travel > 0)
    phi_s = 5.275 * B**-0.3 * sign * slope_along_angle_of_travel**2

    # Effective Heating Number
    epsilon = np.exp(-138 / sigma)
    # Heat of Preignition (BTU/lb)
    Q_ig = 250 + 1116 * M_f

    # Rate of Spread (ft/min)
    R = ((I_R * xi) * (1 + phi_w + phi_s)) / (p_b * epsilon * Q_ig)

    # Take the minimum with 0 because a fire cannot put itself out
    R = np.maximum(R, np.zeros_like(R))

    return R
