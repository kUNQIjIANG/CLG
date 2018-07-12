import numpy as np 
import tensorflow as tf 
from encoder import Encoder
from generator import Generator
from discriminator import Discriminator

class Trainer:
	def __init__(self, hid_units, batch_size, vocab_size, embed_size, c_size, word_embeds, sos_id, eos_id, beam_width):

		self.scope = 'Train'
		self.hid_units = hid_units
		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.c_size = c_size 
		self.word_embed = word_embeds
		self.gene_hid_units = hid_units + c_size
		
		self.encoder = Encoder(self.hid_units)
		self.generator = Generator(self.gene_hid_units,self.batch_size,self.vocab_size,self.c_size,sos_id,eos_id,beam_width)
		self.discriminator = Discriminator(self.hid_units, self.c_size)

		self.build_wake_graph()
		self.build_sleep_graph()
		self.build_infer_graph()

	def build_sleep_graph(self):
		with tf.variable_scope("sleep"):
			self.sleep_input = tf.placeholder(tf.int32,shape = [2*self.batch_size,None], name = 'sleep_input')
			self.sleep_embed = tf.nn.embedding_lookup(self.word_embed, self.sleep_input)
			self.sleep_len = tf.placeholder(tf.int32, shape = [None], name = 'sleep_len')
			self.sleep_labels = tf.placeholder(tf.int32, shape = [None, None], name = 'sleep_labels')

			self.sleep_loss = self.discriminator.discrimi_loss(self.sleep_embed, self.sleep_len, self.sleep_labels)
			self.sleep_train_step = self.optimize(self.sleep_loss)

	def build_wake_graph(self):


		with tf.variable_scope("wake"):

			self.enc_input = tf.placeholder(tf.int32,shape = [self.batch_size,None])
			self.enc_embed = tf.nn.embedding_lookup(self.word_embed, self.enc_input)
			self.enc_len = tf.placeholder(tf.int32, shape = [None])
			#self.enc_len = tf.count_nonzero(self.enc_input, axis = 1)

			self.dec_len = tf.placeholder(tf.int32, shape = [None])

			#self.ini_state = tf.placeholder(tf.float32, shape = [None, None, None], name = "yiyi")
			#self.enc_outputs = tf.placeholder(tf.float32, shape = [None, None, None])
			
			self.dec_tar = tf.placeholder(tf.int32, shape = [self.batch_size, None], name = "jiji")
			self.dec_input = tf.placeholder(tf.int32, shape = [self.batch_size, None], name = "zaza")
			self.dec_embed = tf.nn.embedding_lookup(self.word_embed, self.dec_input)

			#self.dec_len = tf.count_nonzero(self.dec_input)
			self.dec_max_len = tf.reduce_max(self.dec_len)

			z = self.encoder.encode(self.enc_embed,self.enc_len)
			enc_outputs = self.encoder.output_logits(self.enc_embed, self.enc_len)
			
			

			reconst_loss, u, s, logits, self.gen_sen, self.sample_c = self.generator.reconst_loss(self.dec_len, 
																		self.dec_max_len,
																		self.dec_tar, 
																		self.dec_embed,
																		z)
		
		
			logits = tf.reshape(logits, [-1, self.vocab_size])
			logit2word_embeds = tf.matmul(logits, self.word_embed)
			logit_encode = tf.reshape(logit2word_embeds, [self.batch_size,-1,self.embed_size])

			c_loss = self.discriminator.discrimi_loss(logit_encode, self.dec_len, self.sample_c)
		
		
		# length of generated sentence ?

			# 1, force encode focus on unstructure
			# 2, ensure unstructure part of generated sentence unchaged
			new_z = self.encoder.encode(logit_encode, self.dec_len)
			z_loss = self.mutinfo_loss(u,s,new_z)
			
			kl_loss = 0.5 * (tf.reduce_mean(tf.exp(s) + tf.square(u) - s - 1))


			self.generator_loss = reconst_loss + c_loss + z_loss + kl_loss
			self.train_step = self.optimize(self.generator_loss)

	def build_infer_graph(self):
		with tf.variable_scope("infer"):

			self.inf_input = tf.placeholder(tf.int32,shape = [self.batch_size,None])
			self.inf_embed = tf.nn.embedding_lookup(self.word_embed, self.inf_input)
			self.inf_len = tf.placeholder(tf.int32, shape = [None])
			
			inf_z = self.encoder.encode(self.inf_embed,self.inf_len)
			inf_max_len = tf.reduce_max(self.inf_len)
			self.infer_ids = self.generator.infer(inf_z, inf_max_len, self.word_embed)

	def inference(self,sess,inf_input,inf_len):
		infer_ids = sess.run(self.infer_ids,{self.inf_input : inf_input,
											 self.inf_len : inf_len})

		return infer_ids

	def wakeTrain(self,sess,enc_input,enc_len,dec_input,dec_len,dec_tar):

		_, loss, gen_ids, gen_labels = sess.run([self.train_step, self.generator_loss, self.gen_sen, self.sample_c],
									 {self.enc_input : enc_input,
										self.enc_len : enc_len,
										self.dec_input : dec_input,
										self.dec_len : dec_len,
										self.dec_tar : dec_tar})
		print("generator loss: %2f" % loss)
		return gen_ids, gen_labels
		#return gen_sen, sample_c

	def mutinfo_loss(self, z_mean, z_sig, new_z):
		dist = tf.contrib.distributions.MultivariateNormalDiag(z_mean,z_sig,
														validate_args=True)
                                                               
		mutinfo_loss = - dist.log_prob(new_z)              
		return tf.reduce_mean(mutinfo_loss)

	def optimize(self,loss):
		optimizer = tf.train.AdamOptimizer(1e-3)
		gradients, variables = zip(*optimizer.compute_gradients(loss))
		gradients = [None if gradient is None else tf.clip_by_value(gradient,-1.0,1.0) for gradient in gradients]
		train_step = optimizer.apply_gradients(zip(gradients, variables))
		return train_step

	def sleepTrain(self, sess, sleep_input, sleep_len, sleep_labels):
		_, sleep_loss = sess.run([self.sleep_train_step, self.sleep_loss], {self.sleep_input : sleep_input, self.sleep_len : sleep_len, self.sleep_labels : sleep_labels})

		print("sleep loss: %2f " % sleep_loss)

