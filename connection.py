import numpy


class Connection:

    def __init__(self, from_node, to_node, weight, enabled, recursive):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.id_tuple = (from_node.id, to_node.id)
        self.enabled = enabled
        self.recursive = recursive
        if not recursive:
            self.to_node.incoming_signals += 1

    def transmit(self, value):
        if not self.enabled:
            return
        if self.recursive:
            self.to_node.input_value += value
        else:
            self.to_node.set_value(value * self.weight)
