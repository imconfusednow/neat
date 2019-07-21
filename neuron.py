# takes layer, id, output connection

import numpy
import math


class Neuron:

    def __init__(self, layer, id, is_pin, in_sigs):
        self.layer = layer
        self.id = id
        self.is_pin = is_pin
        self.input_value = 0
        self.output_value = 0
        self.connections = []
        self.signals_received = 0
        self.incoming_signals = in_sigs

    def fire(self, do_print):
        if do_print:
            print("////" + str(self.id) + "//////")
            print("input value " + str(self.input_value))
        if self.layer != 0:
            self.output_value = self.activation(self.input_value)
            if do_print:
                print("output value" + str(self.output_value))
        if len(self.connections) == 0:
            return self.output_value
        for i in self.connections:
            i.transmit(self.output_value, do_print)

    def set_value(self, value, do_print):
        if self.layer == 0:
            self.input_value = value
            self.output_value = value
        else:
            self.input_value += value

        self.signals_received += 1
        if self.signals_received == self.incoming_signals:
            self.fire(do_print)

    def activation(self, value):
        return numpy.tanh(value)

    	#return 1 / (1 + math.exp(-value))
    	#return max(0,value)

    def add_connection(self, connection):
        self.connections.append(connection)

    def reset(self):
        self.input_value = 0
        self.signals_received = 0
