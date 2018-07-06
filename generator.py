import tensorflow as tf 
from botModel import botModel
from tensorflow.python.layers.core import Dense

class Generator(botModel):
	def __init__(self,hid_units,batch_size,vocab_size,c_size):
		super().__init__('generator')
		self.hid_units = hid_units
		
		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.c_size = c_size
		self.build_graph()

	def build_graph(self):
		with tf.variable_scope(self.scope):
			self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hid_units, state_is_tuple=True)
			self.sample_c = tf.contrib.distributions.OneHotCategorical(
	            logits=tf.ones([self.batch_size, self.c_size]), dtype=tf.float32).sample()
			
			#disentangle = tf.concat((self.ini_state, self.sample_c), axis=-1)
			#disentangle = tf.reshape(disentangle, [batch_size, self.hid_units+c_size])
			#ini_state = tf.layers.dense(inputs = disentangle, units = self.hid_units,
										#activation = tf.nn.relu)

	def reconst_loss(self, dec_len, dec_max_len, dec_tar, input_embed):
		self.train_helper = tf.contrib.seq2seq.TrainingHelper(inputs = input_embed,
															   sequence_length = dec_len)
		self.initial_state = self.decoder_cell.zero_state(dtype=tf.float32, batch_size = self.batch_size)
		#
		self.train_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.train_helper,
															initial_state = self.initial_state,
															output_layer = Dense(self.vocab_size))
		self.train_outputs, self.final_state, _ = tf.contrib.seq2seq.dynamic_decode(self.train_decoder)
    														
		seq_mask = tf.cast(tf.sequence_mask(dec_len, dec_max_len), tf.float32)
		train_logits = self.train_outputs.rnn_output
		train_ind = self.train_outputs.sample_id
		seq_loss = tf.contrib.seq2seq.sequence_loss(train_logits, dec_tar, seq_mask,
    													average_across_timesteps = False,
    													average_across_batch = True)
		return tf.reduce_mean(seq_loss)
	def outputs(self, dec_len, dec_max_len, dec_tar, input_embed):
		self.train_helper = tf.contrib.seq2seq.TrainingHelper(inputs = input_embed,
															   sequence_length = dec_len)
		self.initial_state = self.decoder_cell.zero_state(dtype=tf.float32, batch_size = self.batch_size)
		#
		self.train_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.train_helper,
															initial_state = self.initial_state,
															output_layer = Dense(self.vocab_size))
		train_outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(self.train_decoder)
    														
		seq_mask = tf.cast(tf.sequence_mask(dec_len, dec_max_len), tf.float32)
		train_logits = train_outputs.rnn_output
		train_ind = train_outputs.sample_id

		return train_logits, train_ind, self.sample_c