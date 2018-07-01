import tensorflow as tf
from botModel import botModel

class Discriminator(botModel):
	def __init__(self,c_size, hid_units):
		self.super.__init__('discriminator')
		self.c_size = c_size
		self.hid_units = hid_units
		self.build_graph()
		
		
	def build_graph(self):
		with tf.variable_scope(self.scope):
			self.enc_inp = tf.placeholders(tf.float32, shape = [None, None, None])
			self.enc_len = tf.placeholders(tf.float32, shape = [None])
			self.true_labels = tf.placeholders(tf.float32, shape = [None, None])
			self.encoder = tf.nn.rnn_cell.BasicLSTMCell(self.hid_units)
			self.enc_output, self.enc_final_state = tf.nn.dynamic_rnn(encoder_cell, enc_input, dtype = tf.float32, sequence_length = enc_len)
			self.pred_c = tf.layers.dense(self.enc_final_state, self.c_size, tf.nn.relu)
			self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.true_labels, logits = self.pred_c))
	
	def predict_c(self, sess, enc_inp, enc_len):
		return sess.run(self.pred_c, {self.enc_inp: enc_inp, self.enc_len: enc_len})

	def discrimi_loss(self, sess, enc_inp, enc_len, true_labels):
		return sess.run(self.cross_entropy, {self.enc_inp : enc_inp, self.enc_len : enc_len, self.true_labels: true_labels})
