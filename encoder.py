import tensorflow as tf
from botModel import botModel

class Encoder(botModel):
	def __init__(self,hid_units, word_embed):
		super().__init__('encoder')
		self.hid_units = hid_units 
		self.word_embed = word_embed
		self.build_graph()

	
	def build_graph(self):
		with tf.variable_scope(self.scope):
			self.enc_input = tf.placeholder(tf.int32,shape = [32,None])
			self.input_embed = tf.nn.embedding_lookup(self.word_embed, self.enc_input)
			self.enc_len = tf.placeholder(tf.int32, shape = [None])
			self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hid_units)
			self.encoder_output, self.encoder_final_state = tf.nn.dynamic_rnn(self.encoder_cell, self.input_embed, 
															dtype = tf.float32, sequence_length = self.enc_len)

	def encode(self,sess,enc_input,enc_len):
		return sess.run(self.encoder_final_state, {self.enc_input : enc_input, self.enc_len : enc_len})

	def output_logits(self,sess,enc_input,enc_len):
		return sess.run(self.encoder_output, {self.enc_input : enc_input, self.enc_len : enc_len})
