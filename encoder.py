import tensorflow as tf

class Encoder(botModel):
	def __init__(self,hid_units):
		super().__init__('encoder')
		self.hid_units = hid_units 
		self.build_graph()
	
	def build_graph(self):
		with tf.variable_scope(self.scope):
			self.enc_input = tf.placeholder(tf.float32,shape = [None,None,None])
			self.enc_len = tf.placeholder(tf.float32, shape = [None])
			self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hid_units)
			self.encoder_output, self.encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, enc_input, dtype = tf.float32, sequence_length = enc_len)

	def encode(self,sess,enc_input,enc_len):
		return sess.run(self.encoder_final_state, {self.enc_input : enc_input, self.enc_len : enc_len})

	def output_logits(self,sess,enc_input,enc_len):
		return sess.run(self.encoder_output, {self.enc_input : enc_input, self.enc_len : enc_len})