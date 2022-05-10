from agent.planning.MCTS import MCTS


class BTAI_3MF:
    """
    The class implementing the Branching Time Active Inference algorithm with
    Multi-Modalities and Multi-Factors.
    """

    def __init__(self, ts, max_planning_steps, exp_const):
        """
        Construct the BTAI_3MF agent.
        :param ts: the temporal slice to be used by the agent.
        :param max_planning_steps: the maximum number of planning iterations.
        :param exp_const: the exploration constant of the Monte-Carlo tree search algorithm.
        """
        self.ts = ts
        self.mcts = MCTS(exp_const)
        self.max_planning_steps = max_planning_steps

    def reset(self, obs):
        """
        Reset the agent to its pre-planning state.
        :param obs: the observation that must be used in the computation of the posterior.
        :return: nothing.
        """
        self.ts.reset()
        self.ts.i_step(obs)

    def step(self):
        """
        Perform planning and action selection.
        :return: the action to execute in the environment.
        """
        for i in range(0, self.max_planning_steps):
            node = self.mcts.select_node(self.ts)
            e_nodes = self.mcts.expansion(node)
            self.mcts.evaluation(e_nodes)
            self.mcts.propagation(e_nodes)
        return max(self.ts.children, key=lambda x: x.visits).action

    def update(self, action, obs):
        """
        Update the agent so that: (1) the root corresponds to the temporal slice reached
        when performing the action passed as parameters, (2) the posterior over hidden
        states takes into account the observation passed as parameters.
        :param action: the action that was executed in the environment.
        :param obs: the observation that was made.
        :return: nothing.
        """
        self.ts = next(filter(lambda x: x.action == action, self.ts.children))
        self.ts.reset()
        self.ts.use_posteriors_as_empirical_priors()
        self.ts.i_step(obs)
