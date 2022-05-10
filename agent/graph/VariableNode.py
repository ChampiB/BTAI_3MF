import torch
from agent.graph.Node import Node


class VariableNode(Node):
    """
    Class representing a variable node in the factor graph.
    """

    def __init__(self, name):
        """
        Construct a node of the factor graph.
        :param name: the node name.
        """
        super().__init__(name, {})

    def compute_message(self, dest_name):
        """
        Compute the message toward the destination node.
        :param dest_name: the name of the destination node.
        :return: the message to the destination node.
        """
        out_msg = None
        for name, message in self.in_messages.items():
            if dest_name == name:
                continue
            out_msg = message if out_msg is None else out_msg * message
        return out_msg
