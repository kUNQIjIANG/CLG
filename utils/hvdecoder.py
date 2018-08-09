import tensorflow as tf
import numpy as np


class HidVecDecoder(tf.contrib.seq2seq.BasicDecoder):
    def __init__(self, cell, helper, initial_state, hid_vec, output_layer=None):
        super().__init__(cell, helper, initial_state, output_layer)
        self.z = hid_vec

    def initialize(self, name=None):
        (finished, first_inputs, initial_state) =  self._helper.initialize() + (self._initial_state,)
        first_inputs = tf.concat([first_inputs, self.z], -1)
        return (finished, first_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        with tf.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
        if self._output_layer is not None:
            cell_outputs = self._output_layer(cell_outputs)
        sample_ids = self._helper.sample(
            time=time, outputs=cell_outputs, state=cell_state)
        (finished, next_inputs, next_state) = self._helper.next_inputs(
            time=time,
            outputs=cell_outputs,
            state=cell_state,
            sample_ids=sample_ids)
        outputs = tf.contrib.seq2seq.BasicDecoderOutput(cell_outputs, sample_ids)
        next_inputs = tf.concat([next_inputs, self.z], -1)
        return (outputs, next_state, next_inputs, finished)
