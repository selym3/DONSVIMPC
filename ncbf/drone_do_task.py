from typing import NamedTuple

import jax.numpy as jnp
import jax
import numpy as np

from robot_planning.helper.convenience import get_drone_kinematics, get_drone_obstacles

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser


def state_to_obs_drone(state: jnp.ndarray, obstacles: jnp.ndarray):
    # [ px pz theta vx vz omega ]
    assert state.shape == (6,)

    # [ px py vx vy r ]
    n_obstacles = obstacles.shape[0]
    assert obstacles.shape == (n_obstacles, 5)
    assert n_obstacles>= 2, "Need at least two obstacles."

    obstacle_positions = obstacles[:, :2]
    obstacle_velocities = obstacles[:, 2:4]
    obstacle_radii = obstacles[:, 4]

    px, pz, theta, vx, vz, omega = state
    pos2d = state[:2]
    vel2d = state[3:5]

    # Compute the distance to the closest two obstacles.
    kin = get_drone_kinematics()

    # sdf
    o_dist = jnp.linalg.norm(pos2d - obstacle_positions) - (obstacle_radii + kin.get_radius())
    # Sort the distances.
    _, two_closest_idx = jax.lax.top_k(-o_dist, k=2) 
    # Take the two closest obstacles.
    o_dist_closest = o_dist[two_closest_idx]
    o_vel_closest = vel2d - obstacle_velocities[two_closest_idx]

    # sincos encoding of theta.
    theta_sincos = jnp.array([jnp.sin(theta), jnp.cos(theta)])

    obs_state = jnp.array([px, pz, vx, vz, omega])
    obs = jnp.concatenate([obs_state, theta_sincos, o_dist_closest, o_vel_closest.flatten()])

    assert obs.shape == (5 + 2 + 2 + 4,)

    return obs


def get_h_components(state, obstacles):
    # [ px pz theta vx vz omega ]
    assert state.shape == (6,)

    n_obstacles = obstacles.shape[0]
    assert obstacles.shape == (n_obstacles, 5)
    assert n_obstacles>= 2, "Need at least two obstacles."

    obstacle_positions = obstacles[:, :2]
    obstacle_velocities = obstacles[:, 2:4]
    obstacle_radii = obstacles[:, 4]

    kin = get_drone_kinematics()

    px, pz, theta, vx, vz, omega = state
    pos2d = state[:2]

    # Obstacles.
    o_dist = jnp.linalg.norm(pos2d - obstacle_positions) - (obstacle_radii + kin.get_radius())
    obs_dist_min = jnp.min(o_dist)

    # negative is safe.
    h_obs = -obs_dist_min

    # Clip the negative side so that we don't spend effort learning regions that are too safe.
    h_obs = jnp.clip(h_obs, -1.0, 0.0)

    # z >= 0, -z <= 0
    h_boundary = -pz  # should be approx [-1, 1] ?

    # is_oob = (theta < -np.pi / 2) | (np.pi / 2 < theta)
    h_drone_angle = jnp.abs(theta) - np.pi / 2
    h_drone_angle = h_drone_angle / (np.pi / 2)  # scale to [-1, 1]

    # Want px >= -4.5
    h_px_left = -(px + 4.5)

    # Just set all the unsafes to 1.
    def f(h_):
        eps = 0.3
        return jnp.where(h_ < 0, h_ - eps, 1.0)

    return {"obs": f(h_obs), "boundary": f(h_boundary), "drone_angle": f(h_drone_angle), "px_left": f(h_px_left)}


def get_h_vector_drone(state, obstacles):
    h_components = get_h_components(state, obstacles)
    h_list = list(h_components.values())
    return jnp.stack(h_list)
