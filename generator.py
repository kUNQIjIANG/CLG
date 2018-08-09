import tensorflow as tf 
from botModel import botModel
from configs import args 
from utils.hvdecoder import HidVecDecoder
from utils.hvbeamdecoder import HidVecBeamDecoder

class Generator(botModel):
	def __init__(self):
		super().__init__('generator')
		self.hid_units = args.hid_units
		
		self.batch_size = args.batch_size
		self.vocab_size = args.vocab_size
		self.c_size = args.c_size
		
		self.beam_width = args.beam_width
		self.sos_id = args.sos_id
		self.eos_id = args.eos_id
		self.build_graph()

	def build_graph(self):
		with tf.variable_scope(self.scope):

			self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hid_units, state_is_tuple=True)
			
			self.outp_layer = tf.layers.Dense(self.vocab_size, tf.nn.softmax,
								 				name = 'outp_layer')
	
	def reparm(self, mean, log_var):
		return mean + tf.exp(0.5 * log_var) * tf.truncated_normal(tf.shape(mean))
	
	def reconst_loss(self, dec_len, dec_max_len, dec_tar, input_embed, u, log_var):
		with tf.variable_scope(self.scope):

			z = self.reparm(u, log_var)
			
			sample_c = tf.contrib.distributions.OneHotCategorical(
	            logits=tf.ones([tf.shape(z)[0], self.c_size]), dtype=tf.float32).sample()
			#sample_c = tf.contrib.distributions.Bernoulli(probs=0.5*tf.ones([self.c_size])).sample([tf.shape(z)[0]])
			
			disentangle = tf.concat((z, sample_c), axis=-1)

			"""
			tile_dise = tf.tile(disentangle,[1,tf.shape(input_embed)[1]])
			exp_dise = tf.expand_dims(tile_dise,1)
			rep_dise = tf.reshape(exp_dise,tf.shape(input_embed))
			dec_input = tf.concat((rep_dise,input_embed),axis=-1)
			"""

			dec_ini_state = tf.contrib.rnn.LSTMStateTuple(disentangle,disentangle)

			self.train_helper = tf.contrib.seq2seq.TrainingHelper(inputs = input_embed,
																   sequence_length = dec_len)
			
			self.train_decoder = HidVecDecoder(self.decoder_cell,
												self.train_helper,
												initial_state = dec_ini_state,
												hid_vec = disentangle,
												output_layer = self.outp_layer)

			self.train_outputs, self.final_state, _ = tf.contrib.seq2seq.dynamic_decode(self.train_decoder)
	    														
			seq_mask = tf.cast(tf.sequence_mask(dec_len, dec_max_len), tf.float32)
			train_logits = self.train_outputs.rnn_output
			train_ind = self.train_outputs.sample_id
			seq_loss = tf.contrib.seq2seq.sequence_loss(train_logits, dec_tar, seq_mask,
	    												average_across_timesteps = False,
	    												average_across_batch = True)
			seq_loss = tf.reduce_sum(seq_loss)
			return seq_loss, train_logits, train_ind, sample_c

	def infer(self, inf_max_len, word_embed, given_c, inf_u, inf_log_var):
		with tf.variable_scope(self.scope, reuse = True):

			inf_z = self.reparm(inf_u,inf_log_var) 

			disentangle = tf.concat((inf_z, given_c), axis=-1)

			"""
			tile_word_embed = tf.tile(word_embed,[tf.shape(inf_u)[0],1])
			exp_word_embed = tf.expand_dims(tile_word_embed,0)
			rep_word_embed = tf.reshape(exp_word_embed,[tf.shape(inf_u)[0],args.vocab_size,args.embed_size])

			tile_dise = tf.tile(disentangle,[1,args.vocab_size])
			exp_dise = tf.expand_dims(tile_dise,1)
			rep_dise = tf.reshape(exp_dise,[tf.shape(disentangle)[0],args.vocab_size,tf.shape(disentangle)[1]])

			batch_vec = tf.concat((rep_word_embed,rep_dise),axis = -1)
			"""

			dec_ini_state = tf.contrib.rnn.LSTMStateTuple(disentangle,disentangle)
			tiled_init_state = tf.contrib.seq2seq.tile_batch(dec_ini_state,
											 multiplier = self.beam_width)

			tiled_hid_vec = tf.tile(tf.expand_dims(disentangle,1),[1,args.beam_width,1])

			beam_decoder = HidVecBeamDecoder(cell=self.decoder_cell,
                                             embedding=word_embed,
                                             start_tokens=tf.fill([tf.shape(inf_u)[0]],self.sos_id),
                                             end_token=self.eos_id,
                                             initial_state= tiled_init_state,
                                             beam_width= self.beam_width,
                                             hid_vec=tiled_hid_vec,
                                             output_layer=self.outp_layer)
			
			infer_outputs, i_final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(beam_decoder,
														maximum_iterations = inf_max_len)

			infer_ids = infer_outputs.predicted_ids[:,:,0]

			return infer_ids
