from agent.inference.TemporalSlice import TemporalSlice
from agent.graph.FactorGraph import FactorGraph


class TemporalSliceBuilder:
    """
    A class representing a temporal slice that can contain several states,
    actions and observations.
    """

    def __init__(self, action_name, n_actions):
        """
        Construct the temporal slice builder.
        :param action_name: the name of the action random variable.
        :param n_actions: the number of values taken by the action random variable.
        """
        if action_name[0:2] != "A_":
            raise Exception("Action name is invalid: name must start with 'A_'.")

        # The name of the action random variable, and the number of actions.
        self.action_name = action_name
        self.n_actions = n_actions

        # The prior belief of each state.
        self.states_prior = {}

        # The parents of each observation, as well as the corresponding
        # likelihood tensor, and prior preferences.
        self.obs_parents = {}
        self.obs_likelihood = {}
        self.obs_prior_pref = {}

        # The parents of each state and the corresponding transition tensor.
        self.states_parents = {}
        self.states_transition = {}

    def add_state(self, rv_name, params):
        """
        Add a latent state to the temporal slice.
        :param rv_name: the name of the state random variable.
        :param params: the parameters of the prior over this state.
        :return: self.
        """
        if rv_name[0:2] != "S_":
            raise Exception("State name is invalid: name must start with 'S_'.")
        if rv_name in self.states_prior.keys():
            raise Exception("State name already exists in the temporal slice.")
        if len(params.shape) != 1:
            raise Exception("Prior parameters must be a 1D-tensor.")
        self.states_prior[rv_name] = params
        return self

    def add_observation(self, rv_name, params, parents):
        """
        Add an observation to the temporal slice.
        :param rv_name: the name of the observation random variable.
        :param params: the parameters of the likelihood mapping for this observation.
        :param parents: a list containing the name of the parent variables of this observation.
        :return: self.
        """
        if len(parents) <= 0:
            raise Exception("An observation must have at least one parent")
        if rv_name[0:2] != "O_":
            raise Exception("Observation name is invalid: name must start with 'O_'.")
        if rv_name in self.obs_likelihood.keys():
            raise Exception("Observation name already exists in the temporal slice.")
        if len(params.shape) != len(parents) + 1:
            raise Exception("Likelihood parameters must be a {}D-tensor.".format(len(parents) + 1))
        self.obs_likelihood[rv_name] = params
        self.obs_parents[rv_name] = parents
        return self

    def add_transition(self, rv_name, params, parents):
        """
        Add a transition mapping to the temporal slice.
        :param rv_name: the name of the state random variable over which the transition is definied.
        :param params: the parameters of the transition mapping for this state.
        :param parents: a list containing the name of the parent variables of this state.
        :return: self.
        """
        if len(parents) <= 0:
            raise Exception("A state must have at least one parent")
        if rv_name[0:2] != "S_":
            raise Exception("State name is invalid: name must start with 'S_'.")
        if rv_name not in self.states_prior.keys():
            raise Exception("State variable has not been created yet.")
        if len(params.shape) != len(parents) + 1:
            raise Exception("Transition parameters must be a {}D-tensor.".format(len(parents) + 1))
        for index, parent in enumerate(parents):
            if parent == self.action_name and params.shape[index + 1] != self.n_actions:
                raise Exception("The action dimension must have a size of {}.".format(self.n_actions))
            if parent != self.action_name and parent not in self.states_prior.keys():
                raise Exception("Variable {} has not been created.".format(parent))
        self.states_transition[rv_name] = params
        self.states_parents[rv_name] = parents
        return self

    def add_preference(self, rv_names, prior_pref):
        """
        Add some prior preferences over a set of observations.
        :param rv_names: the names of the observation random variables.
        :param prior_pref: the prior preference.
        :return: self.
        """
        if type(rv_names) is str:
            rv_names = [rv_names]
        for rv_name in rv_names:
            if rv_name[0:2] != "O_":
                raise Exception("Observation name is invalid: name must start with 'O_'.")
            if rv_name in self.obs_prior_pref.keys():
                raise Exception("There is already some prior preferences over this observation.")
            self.obs_prior_pref[rv_name] = (rv_names, prior_pref)
        return self

    def build(self):
        """
        Build the temporal slice.
        :return: the created temporal slice.
        """
        # Create the factor graph of the temporal slice.
        fg = FactorGraph()
        for state, params in self.states_prior.items():
            fg.add_variable(state)
            fg.add_factor("f_" + state, [state], params)
        for obs in self.obs_likelihood.keys():
            fg.add_variable(obs)
            fg.add_factor("f_" + obs, [obs] + self.obs_parents[obs], self.obs_likelihood[obs])
            fg.add_evidence_placeholder(obs)

        # Create the temporal slice.
        if len(self.obs_likelihood) == 0 or len(self.obs_parents) == 0:
            raise Exception("No observation has been added to the temporal slice.")
        if len(self.states_prior) == 0 or len(self.states_parents) == 0:
            raise Exception("No state has been added to the temporal slice.")
        if len(self.states_prior) != len(self.states_transition):
            raise Exception("The number of transitions must equal the number of states.")
        return TemporalSlice(
            fg, self.n_actions, self.action_name, self.obs_prior_pref,
            self.obs_likelihood, self.states_prior, self.states_transition,
            self.states_parents, self.obs_parents
        )
