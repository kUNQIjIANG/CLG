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
		
		self.optimizer = tf.train.AdamOptimizer(1e-3)
		self.encoder = Encoder(hid_units,c_size)
		self.generator = Generator(hid_units,self.batch_size,self.vocab_size,self.c_size,sos_id,eos_id,beam_width)
		self.discriminator = Discriminator(self.hid_units, self.c_size)

		self.build_placehoders()
		self.build_wake_graph()
		self.build_sleep_graph()
		self.build_infer_graph()
		self.saver = tf.train.Saver()

	def build_placehoders(self):
		
		self.kl_weight = tf.placeholder(tf.float32,shape = [], name = 'kl_weight')
		self.enc_input = tf.placeholder(tf.int32,shape = [None,None], name = 'enc_input')
		self.enc_len = tf.placeholder(tf.int32, shape = [None], name = 'enc_len')
		self.dec_len = tf.placeholder(tf.int32, shape = [None], name = 'dec_len')
		self.dec_tar = tf.placeholder(tf.int32, shape = [None, None], name = "dec_tar")
		self.dec_input = tf.placeholder(tf.int32, shape = [None, None], name = "dec_input")
		self.enc_labels = tf.placeholder(tf.int32, shape = [None, None], name = 'enc_labels')
		self.fake_input = tf.placeholder(tf.int32,shape = [None,None], name = 'fake_input')
		self.fake_len = tf.placeholder(tf.int32, shape = [None], name = 'fake_len')
		self.fake_labels = tf.placeholder(tf.int32, shape = [None, None], name = 'fake_labels')
		self.given_c = tf.placeholder(tf.float32, shape = [None, None], name = 'givenc_c')

	def build_sleep_graph(self):
		
		enc_embed = tf.nn.embedding_lookup(self.word_embed, self.enc_input)

		fake_embed = tf.nn.embedding_lookup(self.word_embed, self.fake_input)

		self.supv_c_logits, self.pre_loss, self.pre_discri_acc = self.discriminator.discrimi_loss(enc_embed, self.enc_len, self.enc_labels)
		self.supv_c = tf.argmax(self.supv_c_logits,axis = 1)
		
		self.pre_train_step = self.optimize(self.pre_loss)

		self.fake_logits, self.fake_loss, self.fake_discri_acc = self.discriminator.discrimi_loss(fake_embed, self.fake_len, self.fake_labels)
		self.entropy_loss = - tf.reduce_sum(tf.nn.log_softmax(self.fake_logits))
		
		self.sleep_acc = (self.pre_discri_acc + self.fake_discri_acc)/2
		
		weight = tf.constant(0.1)
		beta = tf.constant(0.1)
		self.sleep_loss = self.pre_loss + weight * (self.fake_loss + beta * self.entropy_loss)
		self.sleep_train_step = self.optimize(self.sleep_loss)

	def build_wake_graph(self):
		
		enc_embed = tf.nn.embedding_lookup(self.word_embed, self.enc_input)
		dec_embed = tf.nn.embedding_lookup(self.word_embed, self.dec_input)
		# ----------------------VAE Train------------------------------------- #
		self.dec_max_len = tf.reduce_max(self.dec_len)

		_, self.u, self.s = self.encoder.encode(enc_embed,self.enc_len)
		#enc_outputs = self.encoder.output_logits(self.enc_embed, self.enc_len)
		
		self.rec_loss, self.logits, \
		 self.gen_sen, self.sample_c = self.generator.reconst_loss(self.dec_len, 
																	self.dec_max_len,
																	self.dec_tar, 
																	dec_embed,
																	self.u, self.s)
		
		self.kl_loss = 0.5 * tf.reduce_sum(tf.square(self.s) + tf.square(self.u) - 1 - tf.log(tf.square(self.s))) / tf.to_float(tf.shape(enc_embed)[0])

		self.vae_loss = self.rec_loss + self.kl_weight*self.kl_loss
		self.vae_step = self.optimize(self.vae_loss)
		
		# ------------------------- Wake Train --------------------------------- #
		# transfer generator decoder output logits into soft-inputs
		# for distriminator 
		self.logits = tf.nn.softmax(self.logits)
		logits = tf.reshape(self.logits, [-1, self.vocab_size])
		logit2word_embeds = tf.matmul(logits, self.word_embed)
		self.logit_encode = tf.reshape(logit2word_embeds, [tf.shape(enc_embed)[0],-1,self.embed_size])
		
		self.c_logits, self.c_loss, self.syn_acc = self.discriminator.discrimi_loss(self.logit_encode, self.dec_len, self.sample_c)
		self.pred_c = tf.argmax(self.c_logits, axis = -1)
	
	# length of generated sentence ?

		# 1, force encode focus on unstructure
		# 2, ensure unstructure part of generated sentence unchaged
		_, self.new_z, _ = self.encoder.encode(self.logit_encode, self.dec_len)
		self.z_loss = self.mutinfo_loss(self.u,self.s,self.new_z)
		
		#kl_weight = 1 * tf.sigmoid((10/6000)*(tf.to_float(self.step) - tf.constant(6000/2)))

		z_weight = tf.constant(0.1)
		c_weight = tf.constant(0.1)
		self.generator_loss = self.vae_loss + c_weight*self.c_loss + z_weight*self.z_loss
		self.wake_generator_step = self.optimize_with_scope(self.generator_loss,self.generator.scope)
		self.wake_encoder_step = self.optimize_with_scope(self.vae_loss,self.encoder.scope)

	def build_infer_graph(self):
		#with tf.variable_scope("wake",reuse = tf.AUTO_REUSE):

		enc_embed = tf.nn.embedding_lookup(self.word_embed, self.enc_input)
				
		_, infer_u, infer_s = self.encoder.encode(enc_embed,self.enc_len)
		inf_max_len = tf.reduce_max(self.enc_len)

		self.infer_ids = self.generator.infer(inf_max_len, self.word_embed, self.given_c, infer_u, infer_s)

	def inference(self,sess,inf_input,inf_len,given_c):
		infer_ids = sess.run(self.infer_ids,{self.enc_input : inf_input,
											 self.enc_len : inf_len,
											 self.given_c : given_c})

		return infer_ids

	def vaeTrain(self,sess,enc_input,enc_len,dec_input,dec_len,dec_tar,kl_weight):
		_, vae_loss, rec_loss, vae_kl_loss, vae_sen, mean, sig, sample_c  \
							= sess.run([self.vae_step, self.vae_loss, self.rec_loss,
							 			self.kl_loss, self.gen_sen, self.u, self.s,
							 			 self.sample_c],
										 {self.enc_input : enc_input,
											self.enc_len : enc_len,
											self.dec_input : dec_input,
											self.dec_len : dec_len,
											self.dec_tar : dec_tar,
											self.kl_weight : kl_weight})
		#print("generator loss: %2f" % loss)
		return vae_loss, rec_loss, vae_kl_loss, vae_sen, mean, sig, sample_c

	def wakeTrain(self,sess,enc_input,enc_len,dec_input,dec_len,dec_tar,kl_weight):

		_, _, loss, gen_ids, gen_labels, c_loss, z_loss, kl_loss, rec_loss, \
		syn_acc, u, s, pred_c, logit_encode, lgs = sess.run([self.wake_generator_step,
		 				   			self.wake_encoder_step, self.generator_loss,
									self.gen_sen, self.sample_c,
									self.c_loss, self.z_loss,
									self.kl_loss, self.rec_loss,
									self.syn_acc, self.u, self.s,
									self.c_logits, self.logit_encode,
									self.logits],
									{self.enc_input : enc_input,
										self.enc_len : enc_len,
										self.dec_input : dec_input,
										self.dec_len : dec_len,
										self.dec_tar : dec_tar,
										self.kl_weight : kl_weight})
		#print("generator loss: %2f" % loss)
		return gen_ids, gen_labels, c_loss, z_loss, kl_loss, rec_loss, syn_acc, u, s, pred_c, logit_encode, lgs
		#return gen_sen, sample_c

	def mutinfo_loss(self, z_mean, z_sig, new_z):
		dist = tf.contrib.distributions.MultivariateNormalDiag(z_mean,z_sig,
														validate_args=False)
                                                               
		mutinfo_loss = - dist.log_prob(new_z)              
		return tf.reduce_mean(mutinfo_loss)

	def optimize(self,loss):
		
		gvs = self.optimizer.compute_gradients(loss)
		clip_gvs = [(None if grad is None else tf.clip_by_value(grad,-1.0,1.0), var) for grad, var in gvs]
		train_step = self.optimizer.apply_gradients(clip_gvs)
		return train_step

	def optimize_with_scope(self,loss,scope):
		
		var_s = tf.trainable_variables(scope = scope)
		grads = tf.gradients(loss, var_s)
		clip_grads, _ = tf.clip_by_global_norm(grads,5)
		gvs = zip(clip_grads,var_s)
		train_step = self.optimizer.apply_gradients(gvs)
		
		return train_step 

	def preTrain(self, sess, pre_input, pre_len, pre_labels):
		_, pre_loss, pre_acc, supv_c = sess.run([self.pre_train_step,
							 self.pre_loss, self.pre_discri_acc,self.supv_c_logits],
								 {self.enc_input : pre_input,
								  self.enc_len : pre_len,
								  self.enc_labels : pre_labels})

		#print("sleep loss: %2f " % sleep_loss)
		return pre_loss, pre_acc, supv_c

	def sleepTrain(self, sess, real_input, real_len, real_labels, fake_input, fake_len, fake_labels):
		_, sleep_loss, sleep_acc, supv_c = sess.run([self.sleep_train_step,
								  self.sleep_loss, self.sleep_acc,self.supv_c_logits],
								 {self.enc_input : real_input,
								  self.enc_len : real_len,
								  self.enc_labels : real_labels,
								  self.fake_input : fake_input,
								  self.fake_len : fake_len,
								  self.fake_labels : fake_labels})

		#print("sleep loss: %2f " % sleep_loss)
		return sleep_loss, sleep_acc, supv_c 
