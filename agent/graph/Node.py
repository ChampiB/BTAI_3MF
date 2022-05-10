class Node:
    """
    Class representing an abstract node in the factor graph.
    """

    def __init__(self, name, neighbours):
        """
        Construct a node of the factor graph.
        :param name: the node name.
        :param neighbours: the list of neighbours' name.
        """
        self.name = name
        self.neighbours = list(neighbours)
        self.in_messages = {neighbour: None for neighbour in neighbours}

    def compute_message(self, dest_name):
        """
        Compute the message toward the destination node.
        :param dest_name: the name of the destination node.
        """
        raise Exception("Node::compute_message is not implemented")

    def add_neighbours(self, neighbours):
        """
        Add neighbours to the node.
        :param neighbours: the list of neighbours' name.
        :return: nothing.
        """
        self.neighbours += list(neighbours)
        self.in_messages = {
            **self.in_messages,
            **{neighbour: None for neighbour in neighbours}
        }

    def n_neighbours(self):
        """
        Getter.
        :return: the number of neighbours of the node.
        """
        return len(self.in_messages)
