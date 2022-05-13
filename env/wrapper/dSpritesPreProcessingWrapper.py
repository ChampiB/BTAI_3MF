import torch
from torch.nn.functional import one_hot


class dSpritesPreProcessingWrapper:
    """
    Class preforming the pre-processing of the observation coming out
    of the dSprites environment.
    """

    def __init__(self, env, obs_names=None):
        """
        Construct the pre-processor of the dSprites environment.
        :param env: the dSprites environment to be wrapped.
        :param obs_names: the names of each observation.
        """
        self.env = env
        self.n_actions = env.n_actions
        self.obs_names = obs_names if obs_names is not None else \
            ["O_color", "O_shape", "O_scale", "O_orientation", "O_pos_x", "O_pos_y"]
        self.state_names = ["S_color", "S_shape", "S_scale", "S_orientation", "S_pos_x", "S_pos_y"]
        self.s_sizes = self.env.s_sizes.copy()
        self.s_sizes[4] /= self.env.granularity
        self.s_sizes[5] /= self.env.granularity
        self.s_sizes[5] += 1

    def get_reward(self):
        """
        Getter.
        :return: the last reward obtained by the agent.
        """
        return self.env.last_r

    def done(self):
        """
        Getter.
        :return: True if the trial is over, False otherwise.
        """
        return self.env.done()

    def reset(self):
        """
        Reset the environment.
        :return: the initial observation after pre-processing.
        """
        return self.pre_processing(self.env.reset())

    def execute(self, action):
        """
        Execute an action in the environment.
        :param action: the action to be executed.
        :return: the observation made by the agent after pre-processing.
        """
        return self.pre_processing(self.env.execute(action))

    def pre_processing(self, obs):
        """
        Perform the pre-processing on the input observation.
        :param obs: the input observation.
        :return: the observation after pre-processing.
        """
        obs[4] /= self.env.granularity
        obs[5] /= self.env.granularity
        obs_dict = {}
        for i in range(1, obs.size(0)):
            obs_dict[self.obs_names[i]] = one_hot(obs[i].to(torch.int64), self.s_sizes[i])
        return obs_dict

    def current_frame(self):
        """
        Return the current frame (i.e. the current image).
        :return: the current image.
        """
        return self.env.current_frame()

    def render(self):
        """
        Display the current state of the environment as an image.
        :return: nothing.
        """
        self.env.render()

    def a(self, noise=0.001):
        """
        Getter.
        :param noise: specify the amount of noise in the prior beliefs.
        :return: the likelihood mappings for each observation.
        """
        likelihoods = {}
        for i in range(self.s_sizes.size):
            epsilon = noise / (self.s_sizes[i] - 1) if self.s_sizes[i] != 1 else 0
            likelihood = torch.full([self.s_sizes[i], self.s_sizes[i]], epsilon)
            for j in range(self.s_sizes[i]):
                likelihood[j][j] = 1 - noise if self.s_sizes[i] != 1 else 1
            likelihoods[self.obs_names[i]] = likelihood
        return likelihoods

    def b(self, noise=0.001):
        """
        Getter.
        :param noise: specify the amount of noise in the prior beliefs.
        :return: the transitions mappings for each hidden state.
        """
        transitions = {}

        # Generate transitions for which action has no effect.
        for i in range(1, 4):
            transition = torch.full([self.s_sizes[i], self.s_sizes[i]], noise / (self.s_sizes[i] - 1))
            for j in range(self.s_sizes[i]):
                transition[j][j] = 1 - noise
            transitions[self.state_names[i]] = transition

        # Generate transitions for which action has an effect.
        for i in range(4, self.s_sizes.size):
            transition = torch.full([self.s_sizes[i], self.s_sizes[i], self.n_actions], noise / (self.s_sizes[i] - 1))
            for j in range(self.s_sizes[i]):
                cur_state = torch.zeros([self.s_sizes.size])
                cur_state[i] = j * self.env.granularity
                for k in range(self.n_actions):
                    dest_state = self.env.simulate(k, cur_state)
                    dest_id = int(dest_state[i].item() / self.env.granularity)
                    transition[dest_id][j][k] = 1 - noise
            transitions[self.state_names[i]] = transition
        return transitions

    def c(self):
        """
        Getter.
        :return: the prior preferences for each observation.
        """
        preferences = {}

        # Create preferences for observations over which no outcome is preferred.
        for i in [0, 2, 3]:
            preferences[self.obs_names[i]] = torch.full([self.s_sizes[i]], 1 / self.s_sizes[i])

        # Create preferences for the shape and x position of the object.
        preference = torch.zeros([self.s_sizes[4], self.s_sizes[5], self.s_sizes[1]])
        for shape in range(0, 3):
            start_pos = 1 if shape == 0 else 0  # If shape == square then 1 else 0
            end_pos = self.env.n_pos if shape == 0 else self.env.n_pos - 1  # If shape == square then n_pos else n_pos-1
            y_pos = int(32 / self.env.granularity)
            for x_pos in range(start_pos, int(end_pos)):
                preference[x_pos][y_pos][shape] = -5
            if shape == 0:
                preference[0][y_pos][shape] = 5
            else:
                preference[int(self.env.n_pos - 1)][y_pos][shape] = 5
        shape = preference.shape
        preference = torch.softmax(preference.view(-1), dim=0).view(shape)
        preferences[self.obs_names[1] + "_pos_x_y"] = preference

        # Create preferences over the y position.
        noise = 0.6
        preference = torch.full([self.s_sizes[5]], noise / (self.s_sizes[5] - 1))
        preference[self.s_sizes[5] - 1] = 1 - noise
        preferences[self.obs_names[5]] = preference

        return preferences

    def d(self, uniform=False, noise=0.001):
        """
        Getter.
        :param uniform: True if the prior over each state should be uniform, False otherwise.
        :param noise: if uniform is False, specify the amount of noise in the prior beliefs.
        :return: the prior over each hidden state.
        """
        state = self.get_state()
        prior_states = {}
        for i in range(1, self.s_sizes.size):
            if uniform:
                prior = torch.full([self.s_sizes[i]], 1 / self.s_sizes[i])
                prior_states[self.state_names[i]] = prior
            else:
                prior = torch.full([self.s_sizes[i]], noise / (self.s_sizes[i] - 1))
                prior[int(state[i])] = 1 - noise
                prior_states[self.state_names[i]] = prior
        return prior_states

    def get_state(self):
        """
        Getter.
        :return: the state of the environment after accounting for the granularity.
        """
        state = self.env.state
        state[4] /= self.env.granularity
        state[5] /= self.env.granularity
        return state

