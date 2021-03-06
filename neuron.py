# takes layer, id, output connection

import numpy


class Neuron:

    def __init__(self, layer, id, is_pin, in_sigs):
        self.layer = layer
        self.id = id
        self.is_pin = is_pin
        self.input_value = 0
        self.output_value = 0
        self.connections = []
        self.recurrents = []
        self.signals_received = 0
        self.incoming_signals = in_sigs

    def fire(self):
        if self.layer != 0:
            self.output_value = self.activation(self.input_value)
        if len(self.connections) == 0:
            return self.output_value
        for i in self.connections:
            i.transmit(self.output_value)

    def fire_recurrents(self):
        for i in self.recurrents:
            i.transmit(self.output_value)

    def set_value(self, value):
        if self.layer == 0:
            self.input_value += value
            self.output_value = value
        else:
            self.input_value += value

        self.signals_received += 1
        if self.signals_received == self.incoming_signals:
            self.fire()
            self.reset()

    def activation(self, value):
        return numpy.tanh(value)

        # return 1 / (1 + math.exp(-value))
        # return max(0,value)

    def add_connection(self, connection):
        self.connections.append(connection)

    def reset(self):
        self.input_value = 0
        self.signals_received = 0
