import tensorflow as tf
from botModel import botModel
from configs import args

class Discriminator(botModel):
	def __init__(self):
		super().__init__('discriminator')
		self.hid_units = args.hid_units
		self.c_size = args.c_size
		self.build_graph()

	def build_graph(self):
		with tf.variable_scope(self.scope):

			self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hid_units)
			
			#regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
			self.dropout_layer = tf.layers.Dropout(0.7,name = 'dropout_layer')
			self.tanh_layer = tf.layers.Dense(100,tf.tanh,name = 'tanh_layer')
			self.final_layer = tf.layers.Dense(self.c_size,name = 'final_layer')

	def predict_c(self, enc_inp, enc_len):
		with tf.variable_scope(self.scope):
			enc_output, enc_final_state = tf.nn.dynamic_rnn(self.encoder_cell, enc_inp, dtype = tf.float32, sequence_length = enc_len)
			
			dropout = self.dropout_layer(enc_final_state.c)
			tanh = self.tanh_layer(dropout)
			self.pred_c = tf.nn.softmax(self.final_layer(tanh),axis = -1)
			
			return self.pred_c

	def discrimi_loss(self, enc_inp, enc_len, true_labels):
		with tf.variable_scope(self.scope):
			enc_output, enc_final_state = tf.nn.dynamic_rnn(self.encoder_cell, enc_inp, dtype = tf.float32, sequence_length = enc_len)
			
			dropout = self.dropout_layer(enc_final_state.c)
			tanh = self.tanh_layer(dropout)
			self.pred_c = self.final_layer(tanh)
			
			self.cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf.stop_gradient(true_labels), logits = self.pred_c))

			self.correct_prediction = tf.equal(tf.argmax(true_labels, 1), tf.argmax(self.pred_c, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
			
			return self.pred_c, self.cross_entropy, self.accuracy
