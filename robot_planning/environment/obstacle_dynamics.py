
import numpy as np
import jax.numpy as jnp

class ObstacleDynamics:
    def step(step, obstacles, obstacle_velocities):
        pass

class StaticObstacleDynamics(ObstacleDynamics): # Default
    def step(self, obstacles, obstacle_velocities):
        return obstacles, obstacle_velocities


class LinearObstacleDynamics(ObstacleDynamics):
    # obstacles_paths: Nx2x2
    # n obstacles, start point and end point
    def __init__(self, obstacle_paths, dt):
        self.dt = dt
        self.obstacle_paths = obstacle_paths
        # obstacle_paths: (Nx2x2)

    def step(self, obstacles, obstacle_velocities):
        # obstacles: (Nx2) #[may not necessarily be on the obstacle path, that's fine for now]
        # obstacle_velocities: (N, )
        obstacle_paths = self.obstacle_paths

        obstacles = np.copy(obstacles)
        obstacle_velocities = np.copy(obstacle_velocities)

        path_deltas = obstacle_paths[:, 1] - obstacle_paths[:, 0]
        path_norms = np.linalg.norm(path_deltas, axis=1, keepdims=True)
        assert np.all(path_norms > 1e-5), "path norm"
        path_norm_vecs = path_deltas / path_norms
        twod_obstacle_velocities = obstacle_velocities[:, None] * path_norm_vecs
        assert twod_obstacle_velocities.shape == (len(obstacles), 2)

        obstacles += self.dt * twod_obstacle_velocities

        # 2. Boundary Detection using Projection
        # Vector from start to current position
        start_to_obs = obstacles - obstacle_paths[:, 0]
        
        # Dot product: (A dot B) / |B|^2 gives the progress ratio 't'
        # path_norms is |B|, so we square it for |B|^2
        progress = jnp.sum(start_to_obs * path_deltas, axis=1) / (jnp.squeeze(path_norms)**2 + 1e-8)

        # 3. Identify who crossed the line
        past_end = progress > 1.0
        past_start = progress < 0.0
        crossed = past_end | past_start

        # 4. Snap positions to the boundaries if they overshot
        # We use [:, None] to broadcast (N,) booleans to (N, 2) coordinates
        obstacles = jnp.where(past_end[:, None], obstacle_paths[:, 1], obstacles)
        obstacles = jnp.where(past_start[:, None], obstacle_paths[:, 0], obstacles)

        # 5. Flip the scalar velocity for those who crossed
        obstacle_velocities = jnp.where(crossed, -obstacle_velocities, obstacle_velocities)

        return obstacles, obstacle_velocities


        # start_locations = np.squeeze(obstacle_paths[:, 0, :])
        # end_locations = np.squeeze(obstacle_paths[:, 1, :])
        # assert start_locations.shape == (len(obstacles), 2)
        # assert end_locations.shape == (len(obstacles), 2)

        # past_end_bound = ((end_locations < start_locations) & (obstacles < end_locations)) | ((end_locations >= start_locations) & (obstacles >= end_locations))
        # past_end = np.logical_or.reduce(past_end_bound, axis=1)
        # assert past_end.shape == (len(obstacles),)
        # obstacles = obstacles.at[past_end].set(end_locations[past_end])

        # past_start_bound = ((start_locations < end_locations) & (obstacles < start_locations)) | ((start_locations >= end_locations) & (obstacles >= start_locations))
        # past_start = np.logical_or.reduce(past_start_bound, axis=1)
        # assert past_end.shape == (len(obstacles),)
        # obstacles = obstacles.at[past_start].set(start_locations[past_start])

        return obstacles, obstacle_velocities

