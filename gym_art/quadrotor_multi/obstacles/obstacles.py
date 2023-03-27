import copy
import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection


class MultiObstacles:
    def __init__(self, num_obstacles=0, room_dims=np.array([10, 10, 10]), resolution=0.05, obstacle_size=1.0, obst_shape="cube",
                 obst_obs_clip=1.0):
        self.num_obstacles = num_obstacles
        self.room_dims = room_dims
        self.obst_shape = obst_shape
        self.size = obstacle_size
        self.obstacle_radius = obstacle_size / 2.0
        self.pos_arr = []
        self.resolution = resolution
        self.obst_obs_clip = obst_obs_clip

    def reset(self, obs=None, quads_pos=None, pos_arr=None):
        self.pos_arr = copy.deepcopy(np.array(pos_arr))

        quads_sdf_obs = 10 * np.ones((len(quads_pos), 9))
        if self.num_obstacles > 0:
            quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution, obst_obs_clip=self.obst_obs_clip)

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)
        return obs

    def step(self, obs=None, quads_pos=None):
        quads_sdf_obs = 10 * np.ones((len(quads_pos), 9))
        if self.num_obstacles > 0:
            quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution, obst_obs_clip=self.obst_obs_clip)

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)
        return obs

    def collision_detection(self, pos_quads=None):
        quad_collisions = collision_detection(quad_poses=pos_quads[:, :2], obst_poses=self.pos_arr[:, :2],
                                              obst_radius=self.obstacle_radius)

        collided_quads_id = np.where(quad_collisions > -1)[0]
        collided_obstacles_id = quad_collisions[collided_quads_id]
        quad_obst_pair = {}
        for i, key in enumerate(collided_quads_id):
            quad_obst_pair[key] = int(collided_obstacles_id[i])

        return collided_quads_id, quad_obst_pair
