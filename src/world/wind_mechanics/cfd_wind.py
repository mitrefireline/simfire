import math

import numpy as np
import pygame

terrain_features: np.ndarray


class Fluid:
    def __init__(
        self,
        n: int,
        iterations: int,
        scale: int,
        dt: float,
        diffusion: float,
        viscosity: float,
        terrain: np.ndarray,
    ) -> None:
        self.N: int = n  # Width x Height of the Screen
        self.itr = iterations

        self.dt = dt  # Timestep (can be 1 but better if <1 for better behavior)
        self.diff = diffusion  # Controls how vectors and densitys diffuse out
        self.visc = viscosity  # Thickness of the fluid

        self.s = np.zeros((n, n))
        self.density = np.zeros((n, n))

        self.Vx = np.zeros((n, n))
        self.Vy = np.zeros((n, n))

        self.Vx0 = np.zeros((n, n))
        self.Vy0 = np.zeros((n, n))

        self.scale = scale

        self.terrain = terrain
        global terrain_features
        terrain_features = self.terrain

    def addDensity(self, x: int, y: int, amount: float):
        self.density[x][y] += amount

    def addVelocity(self, x: int, y: int, amountX: float, amountY: float):
        self.Vx[x][y] += amountX
        self.Vy[x][y] += amountY

    def step(self):
        diffuse(1, self.Vx0, self.Vx, self.visc, self.dt, self.itr, self.N)
        diffuse(2, self.Vy0, self.Vy, self.visc, self.dt, self.itr, self.N)

        project(self.Vx0, self.Vy0, self.Vx, self.Vy, self.itr, self.N)

        advect(1, self.Vx, self.Vx0, self.Vx0, self.Vy0, self.dt, self.N)
        advect(2, self.Vy, self.Vy0, self.Vx0, self.Vy0, self.dt, self.N)

        project(self.Vx, self.Vy, self.Vx0, self.Vy0, self.itr, self.N)
        diffuse(0, self.s, self.density, self.diff, self.dt, self.itr, self.N)
        advect(0, self.density, self.s, self.Vx, self.Vy, self.dt, self.N)

    def renderD(self, surface):
        for i in range(0, self.N):
            for j in range(0, self.N):
                x = i * self.scale
                y = j * self.scale
                d = self.density[i][j]

                if d > 255:
                    d = 255
                rect = pygame.Rect(x, y, self.scale, self.scale)
                pygame.draw.rect(surface, (d, d, d), rect)
        pygame.display.flip()

    def renderV(self, surface):
        for i in range(0, self.N):
            for j in range(0, self.N):
                x = i * self.scale
                y = j * self.scale
                vx = self.Vx[i][j]
                vy = self.Vy[i][j]

                if not (abs(vx) < 0.1 and abs(vy)) <= 0.1:
                    meanval = int(np.mean([vx, vy]))
                    if meanval < 0:
                        meanval = 0
                    if meanval > 255:
                        meanval = 255
                    pygame.draw.line(
                        surface,
                        [meanval, meanval, meanval, meanval],
                        [x, y],
                        [x + vx, y + vy],
                    )
        pygame.display.flip()


# SUPPORT FUNCTIONS

# mirrors vector values on boundary edges of cells, allows for fluid to cross over
# cells and maintain their vector magnitude


def set_bnd(b: int, x: np.ndarray, N: int):
    # Y Boundaries
    for i in range(1, N - 1):
        x[i][0] = -x[i][1] if b == 2 else x[i][1]
        x[i][N - 1] = -x[i][N - 2] if b == 2 else x[i][N - 2]

    # X Boundaries
    for j in range(1, N - 1):
        x[0][j] = -x[1][j] if b == 1 else x[1][j]
        x[N - 1][j] = -x[N - 2][j] if b == 1 else x[N - 2][j]

    # Handle corners
    x[0][0] = 0.5 * (x[1][0] + x[0][1])
    x[0][N - 1] = 0.5 * (x[1][N - 1] + x[0][N - 2])
    x[N - 1][0] = 0.5 * (x[N - 2][0] + x[N - 1][1])
    x[N - 1][N - 1] = 0.5 * (x[N - 2][N - 1] + x[N - 1][N - 2])

    # Handle Terrain Collisions
    global terrain_features

    # Handle Horizontal Interaction
    if b == 2:
        for row in range(2, N - 2):
            for col in range(2, N - 2):
                # Left to right interaction
                if terrain_features[row][col] == 1.0:
                    x[row][col] = 0.0
                    if terrain_features[row][col - 1] == 0.0:
                        x[row][col - 1] = -1 * x[row][col - 1]
                    if terrain_features[row][col + 1] == 0:
                        x[row][col + 1] = -1 * x[row][col + 1]
    if b == 1:
        for row in range(2, N - 2):
            for col in range(2, N - 2):
                # Left to right interaction
                if terrain_features[row][col] == 1.0:
                    x[row][col] = 0.0
                    if terrain_features[row - 1][col] == 0.0:
                        x[row - 1][col] = -1 * x[row - 1][col]
                    if terrain_features[row + 1][col] == 0.0:
                        x[row + 1][col] = -1 * x[row + 1][col]

    # Sweep Right X
    # for col in range(2, N - 2):
    #     for i in range(2, N - 2):
    #       if terrain_features[i][col] == 1.0:
    #           x[i][col] = -x[i][col - 1] if b == 2 else x[i][col - 1];
    # ## Sweep Left X
    # for col in reversed(range(2, N - 2)):
    #     for i in range(2, N - 2):
    #       if terrain_features[i][col] == 1.0:
    #         x[i][col] =  -x[i][col+1] if b == 2 else x[i][col+1]
    # ## Sweep Down
    # for row in range(2, N - 2):
    #     for j in range(2, N - 2):
    #         if terrain_features[row][j] == 1.0:
    #             x[row][j] = -x[row-1][j] if b == 1 else x[row-1][j];
    # ## Sweep Up
    # for row in reversed(range(2, N - 2)):
    #     for j in range(2, N - 2):
    #         if terrain_features[row][j] == 1.0:
    #             x[row][j] = -x[row+1][j] if b == 1 else x[row+1][j];


def lin_solve(
    b: int, x: np.ndarray, x0: np.ndarray, a: float, c: float, itr: int, N: int
):
    cRecip = 1.0 / c
    for t in range(0, itr):
        for j in range(1, N - 1):
            for i in range(1, N - 1):
                calc = (
                    x0[i][j] + a * (x[i + 1][j] + x[i - 1][j] + x[i][j + 1] + x[i][j - 1])
                ) * cRecip
                global terrain_features
                if terrain_features[i][j] != 1.0:
                    x[i][j] = calc
                else:
                    x[i][j] = 0.0
        set_bnd(b, x, N)


# Precalculates a value and passes everything to lin_solve


def diffuse(
    b: int, x: np.ndarray, x0: np.ndarray, diff: float, dt: float, itr: int, N: int
):
    a = dt * diff * (N - 2) * (N - 2)
    lin_solve(b, x, x0, a, 1 + 6 * a, itr, N)


# Incompressible object, clean up stage for each cell


def project(
    velocX: np.ndarray,
    velocY: np.ndarray,
    p: np.ndarray,
    div: np.ndarray,
    itr: int,
    N: int,
):
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            div[i][j] = (
                -0.5
                * (
                    velocX[i + 1][j]
                    - velocX[i - 1][j]
                    + velocY[i][j + 1]
                    - velocY[i][j - 1]
                )
            ) / N
            p[i][j] = 0

    set_bnd(0, div, N)
    set_bnd(0, p, N)
    lin_solve(0, p, div, 1, 6, itr, N)

    for j in range(1, N - 1):
        for i in range(1, N - 1):
            velocX[i][j] -= 0.5 * (p[i + 1][j] - p[i - 1][j]) * N
            velocY[i][j] -= 0.5 * (p[i][j + 1] - p[i][j - 1]) * N

    set_bnd(1, velocX, N)
    set_bnd(2, velocY, N)


# Responsible for actually moving things around, looks at each cell and grabs its velocity
# then fllows that velocity back in time and sees where it lands.  takes weighted average
# of cells around the spot where it lands, then applies that value to current cell


def advect(
    b: int,
    d: np.ndarray,
    d0: np.ndarray,
    velocX: np.ndarray,
    velocY: np.ndarray,
    dt: float,
    N: int,
):
    dtx = dt * (N - 2)
    dty = dt * (N - 2)

    Nfloat = N - 2
    for j, jfloat in zip(range(1, N - 1), range(1, N - 1)):
        for i, ifloat in zip(range(1, N - 1), range(1, N - 1)):
            tmp1 = dtx * velocX[i][j]
            tmp2 = dty * velocY[i][j]
            x = ifloat - tmp1
            y = jfloat - tmp2

            if x < 0.5:
                x = 0.5
            if x > Nfloat + 0.5:
                x = Nfloat + 0.5
            i0 = math.floor(x)
            i1 = i0 + 1.0

            if y < 0.5:
                y = 0.5
            if y > Nfloat + 0.5:
                y = Nfloat + 0.5
            j0 = math.floor(y)
            j1 = j0 + 1.0

            s1 = x - i0
            s0 = 1.0 - s1
            t1 = y - j0
            t0 = 1.0 - t1

            i0i = int(i0)
            i1i = int(i1)
            j0i = int(j0)
            j1i = int(j1)

            d[i][j] = s0 * (t0 * d0[i0i][j0i] + t1 * d0[i0i][j1i])
            +s1 * (t0 * d0[i1i][j0i] + t1 * d0[i1i][j1i])

    set_bnd(b, d, N)
