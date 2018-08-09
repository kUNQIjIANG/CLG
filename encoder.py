import tensorflow as tf
from botModel import botModel
from configs import args

class Encoder(botModel):
	def __init__(self):
		super().__init__('encoder')
		self.hid_units = args.hid_units 
		self.z_size = args.hid_units - args.c_size
		self.build_graph()

	def build_graph(self):
		with tf.variable_scope(self.scope):
			self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hid_units)
			self.u_layer = tf.layers.Dense(self.z_size,name = 'u_layer')
			self.s_layer = tf.layers.Dense(self.z_size,name = 's_layer')

	def encode(self,input_embed,enc_len):
		with tf.variable_scope(self.scope):
			_, encoder_final_state = tf.nn.dynamic_rnn(self.encoder_cell, input_embed, 
											dtype = tf.float32, sequence_length = enc_len)
			u = self.u_layer(encoder_final_state.c)
			log_var = self.s_layer(encoder_final_state.c)

			return encoder_final_state, u, log_var

	def output_logits(self,input_embed,enc_len):
		with tf.variable_scope(self.scope):
			encoder_output, _ = tf.nn.dynamic_rnn(self.encoder_cell, input_embed, 
											dtype = tf.float32, sequence_length = enc_len)
			return encoder_output;
