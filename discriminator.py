import tensorflow as tf
from botModel import botModel

class Discriminator(botModel):
	def __init__(self,hid_units,c_size):
		super().__init__('discriminator')
		self.hid_units = hid_units
		self.c_size = c_size
		self.build_graph()

	def build_graph(self):
		with tf.variable_scope(self.scope):

			self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hid_units)
			
	def predict_c(self, enc_inp, enc_len):
		with tf.variable_scope(self.scope):
			enc_output, enc_final_state = tf.nn.dynamic_rnn(self.encoder_cell, enc_inp, dtype = tf.float32, sequence_length = enc_len)
			
			self.pred_c = tf.layers.dense(inputs = enc_final_state.c , units = self.c_size, name = 'disen_logits')
			return self.pred_c

	def discrimi_loss(self, enc_inp, enc_len, true_labels):
		with tf.variable_scope(self.scope):
			enc_output, enc_final_state = tf.nn.dynamic_rnn(self.encoder_cell, enc_inp, dtype = tf.float32, sequence_length = enc_len)
			
			self.pred_c = tf.layers.dense(inputs = enc_final_state.c , units = self.c_size, name = 'disen_logits')
			self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = true_labels, logits = self.pred_c))

			self.correct_prediction = tf.equal(tf.argmax(true_labels, 1), tf.argmax(self.pred_c, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
			
			return self.cross_entropy, self.accuracy
