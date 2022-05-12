import random
import torch
from env.viewer.DefaultViewer import DefaultViewer
from data.dSpritesDataset import DataSet
import numpy as np


class dSpritesEnv:
    """
    A class implementing the dSprites environment.
    """

    def __init__(self, granularity=4, repeat=8, dataset_file="./data/dsprites.npz"):
        """
        Construct the dSprites environment.
        :param granularity: the granularity of the x and y positions.
        :param repeat: the number of times an action must be repeated.
        :param dataset_file: path to the file containing the dSprites dataset.
        """
        self.n_actions = 4
        self.granularity = granularity
        self.repeat = repeat

        self.n_pos = 32 / self.granularity
        self.last_r = 0.0
        self.frame_id = 0
        self.max_episode_length = 50
        self.need_reset = False

        self.images, self.s_sizes, self.s_dim, self.s_bases = DataSet.get(dataset_file)
        self.s_dim = self.s_sizes.size
        self.state = torch.zeros(self.s_dim)
        self.s_bases_obs = torch.tensor([0, self.n_pos * (self.n_pos + 1), 0, 0, self.n_pos + 1, 1])
        self.s_bases_img = torch.tensor([737280, 245760, 40960, 1024, 32, 1])

        # Graphical interface
        self.np_precision = np.float64
        self.viewer = None

    def reset_hidden_state(self):
        """
        Reset the hidden state of the environment.
        :return: nothing.
        """
        self.state = torch.zeros(self.s_dim)
        for i in range(0, self.s_dim):
            self.state[i] = random.randint(0, self.s_sizes[i] - 1)

    def reset(self):
        """
        Reset the environment.
        :return: the initial observation.
        """
        self.state = torch.zeros(self.s_dim)
        self.last_r = 0.0
        self.frame_id = 0
        self.need_reset = False
        self.reset_hidden_state()
        return self.state.clone()

    @staticmethod
    def down(state):
        """
        Simulate the execution of the action "down".
        :param state: the state from which the action is executed.
        :return: the new state after performing the action.
        """
        if state[5] + 1 <= 32:
            state[5] += 1
        return state

    @staticmethod
    def up(state):
        """
        Simulate the execution of the action "up".
        :param state: the state from which the action is executed.
        :return: the new state after performing the action.
        """
        if state[5] - 1 >= 0:
            state[5] -= 1
        return state

    @staticmethod
    def left(state):
        """
        Simulate the execution of the action "left".
        :param state: the state from which the action is executed.
        :return: the new state after performing the action.
        """
        if state[4] - 1 >= 0:
            state[4] -= 1
        return state

    @staticmethod
    def right(state):
        """
        Simulate the execution of the action "right".
        :param state: the state from which the action is executed.
        :return: the new state after performing the action.
        """
        if state[4] + 1 < 32:
            state[4] += 1
        return state

    def simulate(self, action, state):
        """
        Simulate the execution of an action in the environment.
        :param action: the action to be simulated.
        :param state: the state from which the action is executed.
        :return: the new state after performing the action.
        """
        actions_fn = [self.down, self.up, self.left, self.right]
        res = state.clone()
        for i in range(0, self.repeat):
            if res[5] >= 32:  # If the agent cross the bottom of the image
                return res    # Then the agent is in an absorbing state
            res = actions_fn[action](res)
        return res

    def execute(self, action):
        """
        Execute an action in the environment.
        :param action: the action to be executed.
        :return: the observation made by the agent.
        """

        # Increase the frame index, that count the number of frames since
        # the beginning of the episode.
        self.frame_id += 1

        # Simulate the action requested by the user.
        self.state = self.simulate(action, self.state)

        # If the object crossed the bottom line, then:
        # compute the reward, generate a new image.
        if self.state[5] >= 32:
            if self.state[1] < 0.5:
                self.last_r = self.compute_square_reward()
            else:
                self.last_r = self.compute_non_square_reward()
            self.need_reset = True

        # Make sure the environment is reset if the maximum number of steps in
        # the episode has been reached.
        if self.frame_id >= self.max_episode_length:
            self.need_reset = True
            self.last_r = -1
        return self.state.clone()

    def compute_square_reward(self):
        """
        Compute the reward obtained by the agent if the shape was a square.
        :return: the reward.
        """
        x_pos = self.state[4].item()
        return (15.0 - x_pos) / 16.0 if x_pos > 15 else (16.0 - x_pos) / 16.0

    def compute_non_square_reward(self):
        """
        Compute the reward obtained by the agent if the shape was not a square.
        :return: the reward.
        """
        x_pos = self.state[4].item()
        return (x_pos - 15.0) / 16.0 if x_pos > 15 else (x_pos - 16.0) / 16.0

    def done(self):
        """
        Getter.
        :return: True if the trial is over, False otherwise.
        """
        return self.need_reset

    def render(self):
        """
        Display the current state of the environment as an image.
        :return: nothing.
        """
        if self.viewer is None:
            self.viewer = DefaultViewer('dSprites', self.last_r, self.current_frame(), frame_id=self.frame_id)
        else:
            self.viewer.update(self.last_r, self.current_frame(), self.frame_id)

    def current_frame(self):
        """
        Return the current frame (i.e. the current observation).
        :return: the current observation.
        """
        if self.state[5] >= 32:
            return None
        image = self.images[self.s_to_index(self.state)].astype(self.np_precision)
        return np.repeat(image, 3, 2) * 255.0

    def s_to_index(self, s):
        """
        Compute the index of the image corresponding to the state sent as parameter.
        :param s: the state whose index must be computed.
        :return: the index.
        """
        return np.dot(s, self.s_bases).astype(int)
