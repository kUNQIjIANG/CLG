import tensorflow as tf 
from botModel import botModel
from tensorflow.python.layers.core import Dense

class Generator(botModel):
	def __init__(self,hid_units,batch_size,vocab_size,c_size):
		super().__init__('generator')
		self.hid_units = hid_units
		self.build_graph(batch_size,vocab_size,c_size)
		

	def build_graph(self,batch_size,vocab_size,c_size):
		self.enc_len = tf.placeholder(tf.float32, shape = [None])
		self.dec_len = tf.placeholder(tf.float32, shape = [None])
		self.dec_max_len = tf.reduce_max(self.dec_len)
		self.ini_state = tf.placeholder(tf.float32, shape = [None, None])
		self.enc_outputs = tf.placeholder(tf.float32, shape = [None, None, None])
		self.dec_tar = tf.placeholder(tf.float32, shape = [None, None])
		self.dec_input = tf.placeholder(tf.float32, shape = [None, None, None])
		#self.emi_layer = tf.layers.dense(vocab_size)
		self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hid_units, reuse = tf.AUTO_REUSE)

		self.sample_c = tf.contrib.distributions.OneHotCategorical(
            logits=tf.ones([batch_size, c_size]), dtype=tf.float32).sample()

		disentangle = tf.concat((self.ini_state, self.sample_c), axis=-1)
		disentangle = tf.reshape(disentangle, [batch_size, self.hid_units+c_size])
		ini_state = tf.layers.dense(inputs = disentangle, units = self.hid_units,
									activation = tf.nn.relu)

		"""
		train_attention_states = self.enc_outputs
		train_attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.hid_units, 
																	train_attention_states,
																	memory_sequence_length = self.enc_len)
		self.train_atten_cell = tf.contrib.seq2seq.AttentionWrapper(self.decoder_cell, train_attention_mechanism,
																	initial_state = ini_state,
																	attention_layer_size = self.hid_units)
		"""



		self.train_helper = tf.contrib.seq2seq.TrainingHelper(inputs = self.dec_input,
															   sequence_length = self.dec_len)
		#self.initial_state = self.decoder_cell.zero_state(dtype=tf.float32, batch_size = batch_size)
		self.train_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.train_helper,
															initial_state = ini_state,
															output_layer = Dense(vocab_size))
		self.train_outputs, self.final_state, _ = tf.contrib.seq2seq.dynamic_decode(self.train_decoder)
    														
		seq_mask = tf.cast(tf.sequence_mask(self.dec_len, self.dec_max_len), tf.float32)
		self.train_logits = self.train_outputs.rnn_output
		self.train_ind = self.train_outputs.sample_id
		self.seq_loss = tf.contrib.seq2seq.sequence_loss(self.train_logits, self.dec_tar, seq_mask,
    													average_across_timesteps = False,
    													average_across_batch = True)

	def reconst_loss(self, sess, enc_len, dec_len, ini_state, enc_outputs, dec_tar, dec_input):
		return	sess.run(self.seq_loss, {self.enc_len : enc_len,
								   self.dec_len : dec_len,
								   self.ini_state : ini_state,
								   self.enc_outputs : enc_outputs,
								   self.dec_iput : dec_input,
								   self.dec_tat : dec_tar})
   	
	def outputs(self, sess, enc_len, dec_len, ini_state, enc_outputs, dec_tar, dec_input):
		return	sess.run([self.train_logits, self.train_ind, self.sample_c], 
										{self.enc_len : enc_len,
									   self.dec_len : dec_len,
									   self.ini_state : ini_state,
									   self.enc_outputs : enc_outputs,
									   self.dec_iput : dec_input,
									   self.dec_tat : dec_tar})
