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

		self.build_pretrain_graph()
		self.build_wake_graph()
		self.build_sleep_graph()
		self.build_infer_graph()

	def build_pretrain_graph(self):
		with tf.variable_scope("pretrain"):
			self.pre_input = tf.placeholder(tf.int32,shape = [None,None], name = 'pre_input')
			self.pre_embed = tf.nn.embedding_lookup(self.word_embed, self.pre_input)
			self.pre_len = tf.placeholder(tf.int32, shape = [None], name = 'pre_len')
			self.pre_labels = tf.placeholder(tf.int32, shape = [None, None], name = 'pre_labels')

			_, self.pre_loss, self.pre_discri_acc = self.discriminator.discrimi_loss(self.pre_embed, self.pre_len, self.pre_labels)
			self.pre_train_step = self.optimize(self.pre_loss)

	def build_sleep_graph(self):
		with tf.variable_scope("sleep", reuse = tf.AUTO_REUSE):
			self.real_input = tf.placeholder(tf.int32,shape = [None,None], name = 'real_input')
			self.real_embed = tf.nn.embedding_lookup(self.word_embed, self.real_input)
			self.real_len = tf.placeholder(tf.int32, shape = [None], name = 'real_len')
			self.real_labels = tf.placeholder(tf.int32, shape = [None, None], name = 'real_labels')

			self.fake_input = tf.placeholder(tf.int32,shape = [None,None], name = 'fake_input')
			self.fake_embed = tf.nn.embedding_lookup(self.word_embed, self.fake_input)
			self.fake_len = tf.placeholder(tf.int32, shape = [None], name = 'fake_len')
			self.fake_labels = tf.placeholder(tf.int32, shape = [None, None], name = 'fake_labels')


			_, self.real_loss, self.real_discri_acc = self.discriminator.discrimi_loss(self.real_embed, self.real_len, self.real_labels)
			self.fake_logits, self.fake_loss, self.fake_discri_acc = self.discriminator.discrimi_loss(self.fake_embed, self.fake_len, self.fake_labels)
			self.entropy_loss = - tf.reduce_sum(tf.nn.log_softmax(self.fake_logits))
			
			self.sleep_acc = (self.real_discri_acc + self.fake_discri_acc)/2
			
			weight = tf.constant(0.1)
			self.sleep_loss = self.real_loss + weight * (self.fake_loss + weight * self.entropy_loss)
			self.sleep_train_step = self.optimize(self.sleep_loss)

	def build_wake_graph(self):


		with tf.variable_scope("wake"):
			self.step = tf.placeholder(tf.int32,shape = [])
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
			
			

			self.reconst_loss, u, s, logits, self.gen_sen, self.sample_c = self.generator.reconst_loss(self.dec_len, 
																		self.dec_max_len,
																		self.dec_tar, 
																		self.dec_embed,
																		z)
		
			# transfer generator decoder output logits into soft-inputs
			# for distriminator 
			logits = tf.reshape(logits, [-1, self.vocab_size])
			logit2word_embeds = tf.matmul(logits, self.word_embed)
			logit_encode = tf.reshape(logit2word_embeds, [self.batch_size,-1,self.embed_size])

			_, self.c_loss, self.syn_acc = self.discriminator.discrimi_loss(logit_encode, self.dec_len, self.sample_c)
		
		
		# length of generated sentence ?

			# 1, force encode focus on unstructure
			# 2, ensure unstructure part of generated sentence unchaged
			self.new_z = self.encoder.encode(logit_encode, self.dec_len)
			self.z_loss = self.mutinfo_loss(u,s,self.new_z)
			
			self.kl_loss = 0.5 * (tf.reduce_sum(tf.exp(s) + tf.square(u) - s - 1)) / tf.to_float(tf.shape(u)[0])
			
			kl_weight = 1 * tf.sigmoid((10/6000)*(tf.to_float(self.step) - tf.constant(6000/2)))

			weight = tf.constant(0.1)
			self.generator_loss = self.reconst_loss + weight*self.c_loss + weight*self.z_loss + kl_weight*self.kl_loss
			self.train_step = self.optimize(self.generator_loss)

	def build_infer_graph(self):
		with tf.variable_scope("infer"):

			self.inf_input = tf.placeholder(tf.int32,shape = [self.batch_size,None])
			self.inf_embed = tf.nn.embedding_lookup(self.word_embed, self.inf_input)
			self.inf_len = tf.placeholder(tf.int32, shape = [None])
			self.given_c = tf.placeholder(tf.float32, shape = [self.batch_size, None])
			
			inf_z = self.encoder.encode(self.inf_embed,self.inf_len)
			inf_max_len = tf.reduce_max(self.inf_len)
			self.infer_ids = self.generator.infer(inf_z, inf_max_len, self.word_embed, self.given_c)

	def inference(self,sess,inf_input,inf_len,given_c):
		infer_ids = sess.run(self.infer_ids,{self.inf_input : inf_input,
											 self.inf_len : inf_len,
											 self.given_c : given_c})

		return infer_ids

	def wakeTrain(self,sess,enc_input,enc_len,dec_input,dec_len,dec_tar,step):

		_, loss, gen_ids, gen_labels, c_loss, z_loss, kl_loss, rec_loss, syn_acc = sess.run([self.train_step, self.generator_loss,
									 self.gen_sen, self.sample_c,
									 self.c_loss, self.z_loss,
									 self.kl_loss, self.reconst_loss,
									 self.syn_acc],
									 {self.enc_input : enc_input,
										self.enc_len : enc_len,
										self.dec_input : dec_input,
										self.dec_len : dec_len,
										self.dec_tar : dec_tar,
										self.step : step})
		#print("generator loss: %2f" % loss)
		return gen_ids, gen_labels, c_loss, z_loss, kl_loss, rec_loss, syn_acc
		#return gen_sen, sample_c

	def mutinfo_loss(self, z_mean, z_sig, new_z):
		dist = tf.contrib.distributions.MultivariateNormalDiag(z_mean,z_sig,
														validate_args=False)
                                                               
		mutinfo_loss = - dist.log_prob(new_z)              
		return tf.reduce_mean(mutinfo_loss)

	def optimize(self,loss):
		optimizer = tf.train.AdamOptimizer(1e-3)
		gradients, variables = zip(*optimizer.compute_gradients(loss))
		gradients = [None if gradient is None else tf.clip_by_value(gradient,-1.0,1.0) for gradient in gradients]
		train_step = optimizer.apply_gradients(zip(gradients, variables))
		return train_step

	def preTrain(self, sess, pre_input, pre_len, pre_labels):
		_, pre_loss, pre_acc = sess.run([self.pre_train_step, self.pre_loss, self.pre_discri_acc],
								 {self.pre_input : pre_input,
								  self.pre_len : pre_len,
								  self.pre_labels : pre_labels})

		#print("sleep loss: %2f " % sleep_loss)
		return pre_loss, pre_acc 


	def sleepTrain(self, sess, real_input, real_len, real_labels, fake_input, fake_len, fake_labels):
		_, sleep_loss, sleep_acc = sess.run([self.sleep_train_step, self.sleep_loss, self.sleep_acc],
								 {self.real_input : real_input,
								  self.real_len : real_len,
								  self.real_labels : real_labels,
								  self.fake_input : fake_input,
								  self.fake_len : fake_len,
								  self.fake_labels : fake_labels})

		#print("sleep loss: %2f " % sleep_loss)
		return sleep_loss, sleep_acc 
