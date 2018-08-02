import gym
from gym import spaces

import numpy as np
import pycuber
import copy


class RLCube(gym.Env):
    def __init__(self):
        self.action_hash = ['L', 'R', 'U', 'D', 'F', 'B']
        self.face_hash = self.action_hash
        self.cube = pycuber.Cube()
        self.debug_face = self.cube.get_face('L')
        #self.solved = copy.deepcopy(self.cube)
        self.solved_as_one_hot = copy.deepcopy(self.get_obs())
        self.action_space = spaces.Discrete(6)
        self.max_episode_steps = 69
        self._max_episode_steps = 69
        self.current_step = 0

        num_faces = 3*3*6
        num_colors = 6
        total_obs_space_size = num_faces*num_colors
        # observation space is a 54 X 6 matrix that is flattened to be 54*6 dims
        # self.observation_space = spaces.Discrete(total_obs_space_size)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_space_size,)) #(total_obs_space_size)

    def step(self, action):
        self.current_step += 1
        self.cube(self.action_hash[action])
        obs = self.get_obs()
        solved_faces = self.solved_as_one_hot == obs
        reward = np.mean(solved_faces)
        if self.cube == pycuber.Cube():
            done = True
        else:
            done = False
        if self.current_step == self.max_episode_steps:
            #self.current_step = 0
            done = True
        return obs, reward, done, dict() #dict(obs_matrix=self.cube_obs_as_matrix())

    def reset(self, step=25):
        print('resetting')
        self.cube = pycuber.Cube()
        for i in range(step):
            self.step(self.action_space.sample())
        self.current_step = 0
        return self.get_obs()

    def get_obs(self):
        obs_matrix = self.cube_obs_as_matrix()
        obs_matrix = obs_matrix.flatten()
        return obs_matrix

    def render(self, mode=None):
        print(self.cube)

    def cube_obs_as_matrix(self):
        cube_matrix = np.zeros((54, 6))
        sq_idx = 0
        for hashz in self.face_hash:
            face = self.cube.get_face(hashz)
            for x in [0, 1, 2]:
                for y in [0, 1, 2]:
                    color_idx = self.hash_color(str(face[x][y]))
                    cube_matrix[sq_idx][color_idx] = 1
                    sq_idx += 1
        return cube_matrix
    
    def hash_color(self, col_str):
        col_str = col_str[1]
        if col_str == 'r':
            return 0
        elif col_str == 'o':
            return 1
        elif col_str == 'y':
            return 2
        elif col_str == 'w':
            return 3
        elif col_str == 'g':
            return 4
        elif col_str == 'b':
            return 5
        raise ValueError('invalid color string')


if __name__ == '__main__':
    test_cube = RLCube()
    test_cube.reset()
    import time
    t = time.time()
    for _ in range(1000):
        act = test_cube.action_space.sample()
        obs, rew, done, extra_dict = test_cube.step(act)
        #obs_matrix = extra_dict['obs_matrix']
    t2 = time.time()
    print((t2 - t) / 1.0)

