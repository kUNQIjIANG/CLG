import numpy as np
import tensorflow as tf
from encoder import Encoder
from generator import Generator
from discriminator import Discriminator
from configs import args

class Trainer:
	def __init__(self):

		self.scope = 'Train'
		self.hid_units = args.hid_units
		self.batch_size = args.batch_size
		self.vocab_size = args.vocab_size
		self.embed_size = args.embed_size
		self.c_size = args.c_size
		self.word_embed = tf.get_variable("word_embeds",[args.vocab_size, args.embed_size])

		self.optimizer = tf.train.AdamOptimizer(1e-3)
		self.encoder = Encoder()
		self.generator = Generator()
		self.discriminator = Discriminator()

		self.build_placehoders()
		self.build_wake_graph()
		self.build_sleep_graph()
		self.build_infer_graph()
		self.saver = tf.train.Saver(max_to_keep = 1)

	def build_placehoders(self):

		self.step = tf.placeholder(tf.float32,shape = [],name = 'step')
		self.temp_step = tf.placeholder(tf.float32,shape = [],name = 'temp_step')
		#self.kl_weight = tf.placeholder(tf.float32,shape = [], name = 'kl_weight')
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

	def reparm(self, mean, log_var):

		return mean + tf.exp(0.5 * log_var) * tf.truncated_normal(tf.shape(mean))

	def draw_c(self,z):
		
		return tf.contrib.distributions.OneHotCategorical(
            logits=tf.ones([tf.shape(z)[0], self.c_size]), dtype=tf.float32).sample()

	def build_sleep_graph(self):

		enc_embed = tf.nn.embedding_lookup(self.word_embed, self.enc_input)

		fake_embed = tf.nn.embedding_lookup(self.word_embed, self.fake_input)

		self.supv_c_logits, self.pre_loss, self.pre_discri_acc = self.discriminator.discrimi_loss(enc_embed, self.enc_len, self.enc_labels)
		self.supv_c = tf.argmax(self.supv_c_logits,axis = 1)

		self.pre_train_step, _ = self.optimize(self.pre_loss)

		self.fake_logits, self.fake_loss, self.fake_discri_acc = self.discriminator.discrimi_loss(fake_embed, self.fake_len, self.fake_labels)
		self.entropy_loss = - tf.reduce_sum(tf.nn.log_softmax(self.fake_logits))

		self.sleep_acc = (self.pre_discri_acc + self.fake_discri_acc)/2

		weight = tf.constant(0.1)
		beta = tf.constant(0.1)
		self.sleep_loss = self.pre_loss + weight * (self.fake_loss + beta * self.entropy_loss)
		self.sleep_train_step, _ = self.optimize(self.sleep_loss)

	def build_wake_graph(self):

		enc_embed = tf.nn.embedding_lookup(self.word_embed, self.enc_input)
		dec_embed = tf.nn.embedding_lookup(self.word_embed, self.dec_input)
		# ----------------------VAE Train------------------------------------- #
		self.dec_max_len = tf.reduce_max(self.dec_len)

		_, self.u, self.log_var = self.encoder.encode(enc_embed,self.enc_len)
		#enc_outputs = self.encoder.output_logits(self.enc_embed, self.enc_len)
		real_z = self.reparm(self.u, self.log_var)
		self.sample_c = self.draw_c(real_z)

		disentangle = tf.concat((real_z, self.sample_c), axis=-1)
		self.rec_loss, self.logits, \
		 self.gen_sen  = self.generator.reconst_loss(self.dec_len,
													self.dec_max_len,
													self.dec_tar,
													dec_embed,
													disentangle)

		self.kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.log_var) + tf.square(self.u) - 1 - self.log_var) / tf.to_float(tf.shape(enc_embed)[0])

		self.kl_w = 1 * tf.sigmoid((10/6000)*(tf.to_float(self.step) - tf.constant(6000/2)))

		self.vae_loss = self.rec_loss + self.kl_w * self.kl_loss
		self.vae_step, self.vae_gvs = self.optimize(self.vae_loss)

		# ------------------------- Wake Train --------------------------------- #
		# transfer generator decoder output logits into soft-inputs
		# for distriminator
		self.temp = self.temp_fn(self.temp_step)
		self.logits = tf.nn.softmax(self.logits[:,:-1,:]/self.temp)
		logits = tf.reshape(self.logits, [-1, self.vocab_size])
		logit2word_embeds = tf.matmul(logits, self.word_embed)
		self.logit_encode = tf.reshape(logit2word_embeds, [tf.shape(enc_embed)[0],-1,self.embed_size])

		self.c_logits, self.c_loss, self.syn_acc = self.discriminator.discrimi_loss(self.logit_encode, self.dec_len, self.sample_c)
		self.pred_c = tf.argmax(self.c_logits, axis = -1)

	# length of generated sentence ?

		# 1, force encode focus on unstructure
		# 2, ensure unstructure part of generated sentence unchaged
		_, new_u, new_log_var = self.encoder.encode(self.logit_encode, self.dec_len)
		self.z_loss = self.mutinfo_loss(new_u,new_log_var,self.u)

		z_weight = tf.constant(0.1)
		c_weight = tf.constant(0.1)
		self.generator_loss = self.vae_loss + c_weight*self.c_loss + z_weight*self.z_loss
		self.wake_generator_step, self.gene_gvs = self.optimize_with_scope(self.generator_loss,self.generator.scope)
		self.wake_encoder_step, self.enc_gvs = self.optimize_with_scope(self.vae_loss,self.encoder.scope)

	def build_infer_graph(self):

		enc_embed = tf.nn.embedding_lookup(self.word_embed, self.enc_input)

		_, inf_u, inf_log_var = self.encoder.encode(enc_embed,self.enc_len)
		inf_z = self.reparm(inf_u,inf_log_var)
		disentangle = tf.concat((inf_z, self.given_c), axis=-1)

		inf_max_len = tf.reduce_max(self.enc_len)

		self.infer_ids = self.generator.infer(inf_max_len, self.word_embed, disentangle)

	def inference(self,sess,inf_input,inf_len,given_c):
		infer_ids = sess.run(self.infer_ids,{self.enc_input : inf_input,
											 self.enc_len : inf_len,
											 self.given_c : given_c})

		return infer_ids

	def vaeTrain(self,sess,enc_input,enc_len,dec_input,dec_len,dec_tar,step):
		_, vae_loss, rec_loss, vae_kl_loss, vae_sen, mean, log_var, sample_c, kl_w \
							= sess.run([self.vae_step, self.vae_loss, self.rec_loss,
							 			self.kl_loss, self.gen_sen, self.u, self.log_var,
							 			 self.sample_c,self.kl_w],
										 {self.enc_input : enc_input,
											self.enc_len : enc_len,
											self.dec_input : dec_input,
											self.dec_len : dec_len,
											self.dec_tar : dec_tar,
											self.step : step})
		#print("generator loss: %2f" % loss)
		return vae_loss, rec_loss, vae_kl_loss, vae_sen, mean, log_var, sample_c, kl_w

	def wakeTrain(self,sess,enc_input,enc_len,dec_input,dec_len,dec_tar,step,temp_step):

		_, _, loss, gen_ids, gen_labels, c_loss, z_loss, kl_loss, rec_loss, \
		syn_acc, u, s, pred_c, logit_encode, lgs, kl_w, temp  = sess.run([self.wake_generator_step,
		 				   			self.wake_encoder_step, self.generator_loss,
									self.gen_sen, self.sample_c,
									self.c_loss, self.z_loss,
									self.kl_loss, self.rec_loss,
									self.syn_acc, self.u, self.log_var,
									self.c_logits, self.logit_encode,
									self.logits, self.kl_w, self.temp],
									{self.enc_input : enc_input,
										self.enc_len : enc_len,
										self.dec_input : dec_input,
										self.dec_len : dec_len,
										self.dec_tar : dec_tar,
										self.step : step,
										self.temp_step : temp_step})
		#print("generator loss: %2f" % loss)
		return gen_ids, gen_labels, c_loss, z_loss, kl_loss, rec_loss, \
		 syn_acc, u, s, pred_c, logit_encode, lgs, kl_w, temp
		#return gen_sen, sample_c

	def mutinfo_loss(self, z_mean, log_var, new_z):
		dist = tf.contrib.distributions.MultivariateNormalDiag(z_mean,tf.exp(log_var),
														validate_args=True)

		mutinfo_loss = - dist.log_prob(new_z)
		return tf.reduce_sum(mutinfo_loss/tf.to_float(tf.shape(new_z)[1]))

	def optimize(self,loss):

		var_s = tf.trainable_variables()
		grads = tf.gradients(loss, var_s)
		clip_grads, _ = tf.clip_by_global_norm(grads, args.clip_norm)
		clip_gvs = zip(clip_grads, var_s)
		train_step = self.optimizer.apply_gradients(clip_gvs)

		return train_step, var_s

	def optimize_with_scope(self,loss,scope):

		var_s = tf.trainable_variables(scope = scope)
		grads = tf.gradients(loss, var_s)
		clip_grads, _ = tf.clip_by_global_norm(grads,args.clip_norm)
		clip_gvs = zip(clip_grads,var_s)
		train_step = self.optimizer.apply_gradients(clip_gvs)

		return train_step, clip_grads

	def preTrain(self, sess, pre_input, pre_len, pre_labels):
		_, pre_loss, pre_acc, supv_c = sess.run([self.pre_train_step,
							 self.pre_loss, self.pre_discri_acc,self.supv_c_logits],
								 {self.enc_input : pre_input,
								  self.enc_len : pre_len,
								  self.enc_labels : pre_labels})

		#print("sleep loss: %2f " % sleep_loss)
		return pre_loss, pre_acc, supv_c

	def temp_fn(self,temp_step):
		inv_sigmoid = lambda x: 1 / (1 + tf.exp(x))
		return args.temp_anneal_max * inv_sigmoid((10 / args.temp_anneal_bias) * (
		    tf.to_float(temp_step) - tf.constant(args.temp_anneal_bias / 2)))


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
