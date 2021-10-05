from math import exp, tan


# Set constant values for each tile
# Each tile is 1x1 (ft)
tile_length = 1
tile_area = tile_length**2
# The fuel in each tile is 1ft high
# Delta is fuel bed depth (ft)
tile_height = delta = 1
tile_volume = tile_area * tile_height
# Oven-dry Fuel Load (lb/ft^2)
w_0 = 1 # I'm not really sure what a reasonable value is
# Slope steepnes of tile
phi = 0

# Environmental factors
# Moisture Content
M_f = 0.5
# Wind Velocity at midflame height (ft/min)
wind_mph = 10
U = 88 * wind_mph

# Fuel particle values
# Constants (Could be changed, but paper uses constants)
# These constants relate to the fuel particles themselves
# Low Heat Content (BTU/lb)
h = 8000
# Total Mineral Content
S_T = 0.0555
# Effective Mineral Content
S_e = 0.01
# Oven-dry Particle Density (lb/ft^3)
p_p = 32

# Fuel array values
# These values are determined by the environment/fuel
# Surface-area-to-volume ratio (ft^2/ft^3)
sigma = tile_area / tile_volume
# Mineral Damping Coefficient
eta_S = max(0.174 * S_e**-0.19, 1)
# Moisture Damping Coefficient
r_M = max(M_f / M_x, 1)
eta_M = 1 - 2.59*r_M + 5.11*r_M**2 - 3.52*r_M**3
# Net Fuel Load (lb/ft^2)
w_n = w_0 * (1 - S_T)
# Oven-dry Bulk Density (lb/ft^3)
p_b = w_0 / delta
# Packing Ratio
B = p_b / p_p
# Optimum Packing Ratio
B_op = 3.348 * sigma**-0.8189
# Maximum Reaction Velocity (1/min)
gamma_prime_max = sigma**1.5 / (495 + 0.0594*sigma**1.5)
A = 133 * sigma**-0.7913
# Optimum Reaction Velocity (1/min)
gamma_prime = gamma_prime_max * (B/B_op)**A * exp(A*(1-B/B_op))
# Reaction Intensity (BTU/ft^2-min)
I_R = gamma_prime * w_n * h * eta_M * eta_S
# Propagating Flux Ratio
xi = exp((0.792 + 0.681*sigma**0.5) * (B + 0.1)) / (192 + 0.25*sigma)
# Wind Factor
c = 7.47 * exp(-0.133*sigma**0.55)
b = 0.02526 * sigma**0.54
e = 0.715 * exp(-3.59e-4 * sigma)
phi_w = c * U**b * (B/B_op)**-e
# Slope Factor
phi_s = 5.275 * B**-0.3 * tan(phi)**2
# Effective Heating Number
epsilon = exp(-138/sigma)
# Heat of Preignition (BTU/lb)
Q_ig = 250 + 1116*M_f
# Rate of Spread (ft/min)
R = (I_R * xi * (1+phi_w+phi_s)) / (p_b * epsilon * Q_ig)