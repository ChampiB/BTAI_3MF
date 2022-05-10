from agent.graph.Node import Node
from agent.inference.Operators import Operators


class FactorNode(Node):
    """
    Class representing a factor node in the factor graph.
    """

    def __init__(self, name, neighbours, params):
        """
        Construct a factor node.
        :param name: the node's name.
        :param neighbours: the list of neighbours' name.
        :param params: the parameters of the factor graph.
        """
        super().__init__(name, neighbours)
        self.params = params

    def compute_message(self, dest_name):
        """
        Compute the message toward the destination node.
        :param dest_name: the name of the destination node.
        """
        out_msg = self.params
        if out_msg is None:
            raise Exception("In FactorNode::compute_message, {}.param is None.".format(self.name))
        for name in reversed(self.neighbours):
            if dest_name == name:
                continue
            message = self.in_messages[name]
            i = self.neighbours.index(name)
            out_msg = Operators.average(out_msg, message, [i])
        return out_msg
