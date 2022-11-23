from math import cos, sin, pi
from matplotlib import pyplot as plt

GRAV = 6.674e-11  # Gravitational Constant
DELTA = 50000     # Time increment


class Planet:

    def __init__(self,
                 name: str,
                 mass: float,
                 position: tuple[float, float] = None,
                 max_speed: float = None,
                 velocity: tuple[float, float] = (0, 0),
                 polar_position: tuple[float, float] = None):

        self.name = name
        self.mass = mass
        self.max_speed = max_speed

        if position:
            self.position = position
        elif polar_position:
            print(polar_position)
            self.position = self.cartesian_from_polar(polar_position)

        if max_speed:
            self.velocity = self.velocity_from_speed()
        else:
            self.velocity = velocity

        self.acceleration = (0, 0)
        self.force = (0, 0)

    def cartesian_from_polar(self, polar_coords: tuple[float, float]) -> tuple[float, float]:
        r, theta = polar_coords
        x = r*cos(theta)
        y = r*sin(theta)
        return x, y

    def velocity_from_speed(self) -> tuple[float, float]:

        v = self.max_speed
        x, y = self.position

        direction = -1 if x<0 or y<0 else 1

        v_x = direction * -((v**2 * y**2) / (x**2 + y**2))**0.5
        v_y = direction * ((v**2 * x**2) / (x**2 + y**2))**0.5

        return v_x, v_y

    def calc_force(self, p: 'Planet') -> tuple[float, float]:

        x1_pos, y1_pos = self.position
        x2_pos, y2_pos = p.position

        x_dis = x2_pos-x1_pos
        y_dis = y2_pos-y1_pos

        r_sqr = abs(x_dis**2 + y_dis**2)+1e-308
        f_mag = (GRAV*self.mass*p.mass)/r_sqr

        fx = (f_mag*x_dis)/(r_sqr**(1/2))
        fy = (f_mag*y_dis)/(r_sqr**(1/2))

        return fx, fy

    def calc_total_force(self, planets: list['Planet']) -> tuple[float, float]:

        forces = [self.calc_force(p) for p in planets]

        fx_tot = sum([fx for fx,fy in forces])
        fy_tot = sum([fy for fx,fy in forces])

        return fx_tot, fy_tot

    def calc_acceleration(self, force: tuple[float, float]) -> tuple[float, float]:

        fx, fy = force

        ax = fx/self.mass
        ay = fy/self.mass

        return ax, ay

    def calc_velocity(self, acceleration: tuple[float, float]) -> tuple[float, float]:
        ax, ay = acceleration
        vx, vy = self.velocity
        vx = vx+(DELTA*ax)
        vy = vy+(DELTA*ay)
        return vx, vy

    def calc_position(self, velocity: tuple[float, float]) -> tuple[float, float]:
        x, y = self.position
        vx, vy = velocity
        x = x+(DELTA*vx)
        y = y+(DELTA*vy)
        return x, y

    def update_planet(self, planets: list['Planet']) -> None:
        self.force = self.calc_total_force(planets)
        self.acceleration = self.calc_acceleration(self.force)
        self.velocity = self.calc_velocity(self.acceleration)
        self.position = self.calc_position(self.velocity)


class System:

    def __init__(self, planets: list[Planet] = []):

        self.planets = planets
        self.planet_traj = {}

    def add_planet(self, planet: Planet) -> None:
        self.planets.append(planet)

    def check_name(self):
        pass

    def run(self, time: int = 50000):
        planet_traj = {planet.name: ([], []) for planet in self.planets}
        run_time = time
        for t in range(0, run_time):
            for planet in self.planets:
                other_planets = [p for p in self.planets if p != planet]
                planet.update_planet(other_planets)
                planet_traj[planet.name][0].append(planet.position[0])
                planet_traj[planet.name][1].append(planet.position[1])

        self.planet_traj = planet_traj

    def plot(self, x_lim=None, y_lim=None):

        plt.figure(figsize=(18, 18))
        if x_lim:
            plt.xlim(x_lim)
        if y_lim:
            plt.ylim(y_lim)

        for planet_name, planet_traj in self.planet_traj.items():
            plt.plot(*planet_traj, label=planet_name)
        plt.legend()
        plt.show()
