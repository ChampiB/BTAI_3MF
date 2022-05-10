from agent.graph.VariableNode import VariableNode
from agent.graph.FactorNode import FactorNode


class FactorGraph:
    """
    Class representing a factor graph.
    """

    def __init__(self):
        """
        Construct an empty factor graph.
        """
        self.nodes = {}

    def __getitem__(self, index):
        """
        Get the node corresponding to the index.
        :param index: the node index.
        :return: the node.
        """
        return self.nodes[index]

    def add_variable(self, var_name):
        """
        Add a variable to the factor graph.
        :param var_name: the variable name.
        :return: nothing.
        """
        self.nodes[var_name] = VariableNode(var_name)

    def add_factor(self, factor_name, neighbours, params):
        """
        Add a factor to the factor graph.
        :param factor_name: the factor name.
        :param neighbours: the list of neighbours' name.
        :param params: the factor parameters.
        :return: nothing.
        """
        self.nodes[factor_name] = FactorNode(factor_name, neighbours, params)
        for neighbour in neighbours:
            self.nodes[neighbour].add_neighbours([factor_name])

    def add_evidence_placeholder(self, obs_name):
        """
        Add a factor containing the evidenc of an observed variable.
        :param obs_name: the observed variable name.
        :return: nothing.
        """
        self.add_factor("e_" + obs_name, [obs_name], None)

    def set_evidence(self, obs_name, evidence):
        """
        Set the parameters of the factor representing the evidenc of an observed variable.
        :param obs_name: the observed variable name.
        :param evidence: factor's parameter encoding the evidence.
        :return: nothing.
        """
        # Check if the observation name is valid.
        if "e_" + obs_name not in self.nodes.keys():
            print("Warning: e_{} is not in the factor graph's nodes.".format(obs_name))
            return

        # Set the evidence.
        self.nodes["e_" + obs_name].params = evidence

    def reset_messages(self):
        """
        Reset all the messages of the factor graph.
        :return: nothing.
        """
        for node in self.nodes.values():
            for neighbour in node.in_messages.keys():
                node.in_messages[neighbour] = None

    def leaf_nodes(self):
        """
        Getter.
        :return: the list of all leaf nodes in the factor graph.
        """
        return filter(lambda node: node.n_neighbours() == 1, self.nodes.values())

    def state_nodes(self):
        """
        Getter.
        :return: the list of all nodes representing hidden states in the factor graph.
        """
        return filter(lambda node: node.name[0:2] == "S_", self.nodes.values())
