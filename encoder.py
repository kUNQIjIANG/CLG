import tensorflow as tf
from botModel import botModel

class Encoder(botModel):
	def __init__(self,hid_units):
		super().__init__('encoder')
		self.hid_units = hid_units 
		self.build_graph()

	def build_graph(self):
		with tf.variable_scope(self.scope):
			self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hid_units)

	def encode(self,input_embed,enc_len):
		_, encoder_final_state = tf.nn.dynamic_rnn(self.encoder_cell, input_embed, 
															dtype = tf.float32, sequence_length = enc_len)
		return encoder_final_state;
	def output_logits(self,input_embed,enc_len):
		encoder_output, _ = tf.nn.dynamic_rnn(self.encoder_cell, input_embed, 
															dtype = tf.float32, sequence_length = enc_len)
		return encoder_output;