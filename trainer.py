import numpy as np 
import tensorflow as tf 
from encoder import Encoder
from generator import Generator
from discriminator import Discriminator

class Trainer:
	def __init__(self, hid_units, batch_size, vocab_size, embed_size, c_size, word_embeds):

		self.scope = 'Train'
		self.hid_units = hid_units
		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.c_size = c_size 
		self.word_embed = word_embeds
		self.build_graph()
             
	def build_graph(self):

		self.encoder = Encoder(self.hid_units)
		self.generator = Generator(self.hid_units,self.batch_size,self.vocab_size,self.c_size)
		self.discriminator = Discriminator(self.hid_units, self.c_size)

		with tf.variable_scope(self.scope):

			self.enc_input = tf.placeholder(tf.int32,shape = [32,None])
			self.enc_embed = tf.nn.embedding_lookup(self.word_embed, self.enc_input)
			self.enc_len = tf.placeholder(tf.int32, shape = [None])


			self.enc_len = tf.placeholder(tf.float32, shape = [None])
			self.dec_len = tf.placeholder(tf.int32, shape = [None])
			self.dec_max_len = tf.reduce_max(self.dec_len)
			
			#self.ini_state = tf.placeholder(tf.float32, shape = [None, None, None], name = "yiyi")
			#self.enc_outputs = tf.placeholder(tf.float32, shape = [None, None, None])
			
			self.dec_tar = tf.placeholder(tf.int32, shape = [self.batch_size, None], name = "jiji")
			self.dec_input = tf.placeholder(tf.int32, shape = [self.batch_size, None], name = "zaza")
			self.dec_embed = tf.nn.embedding_lookup(self.word_embed, self.dec_input)


			self.enc_inp = tf.placeholder(tf.float32, shape = [self.batch_size, None, self.embed_size])
			self.enc_len = tf.placeholder(tf.int32, shape = [None])
			self.true_labels = tf.placeholder(tf.int32, shape = [None, None])

			#enc_input = tf.cast(enc_input, tf.int32)
			#enc_embed = tf.nn.embedding_lookup(self.word_embeds, enc_input)
			
			z = self.encoder.encode(self.enc_embed,self.enc_len)
			enc_outputs = self.encoder.output_logits(self.enc_embed, self.enc_len)
			
			

			reconst_loss = self.generator.reconst_loss(self.dec_len, self.dec_max_len, self.dec_tar, self.dec_embed)
			
			logits, gen_sen, sample_c = self.generator.outputs(self.dec_len, self.dec_max_len, self.dec_tar, self.dec_embed)
		
			logits = tf.reshape(logits, [-1, self.vocab_size])
			logit2word_embeds = tf.matmul(logits, self.word_embed)
			logit_encode = tf.reshape(logit2word_embeds, [self.batch_size,-1,self.embed_size])

			c_loss = self.discriminator.discrimi_loss(logit_encode, self.dec_len, sample_c)
		
		
		# length of generated sentence ?

			self.generator_loss = reconst_loss + c_loss
			self.train_step = self.optimize(self.generator_loss)

	def wakeTrain(self,sess,enc_input,enc_len,dec_input,dec_len,dec_tar):

		_, loss = sess.run([self.train_step, self.generator_loss],
									 {self.enc_input : enc_input,
										self.enc_len : enc_len,
										self.dec_input : dec_input,
										self.dec_len : dec_len,
										self.dec_tar : dec_tar})
		print(loss)
		"""

		# 1, force encode focus on unstructure
		# 2, ensure unstructure part of generated sentence unchaged
		new_z = self.encoder.encode(sess, gen_sen, dec_len)
		z_loss = tf.nn.l2_loss(z.c - new_z.c)

		c_loss = tf.convert_to_tensor(reconst_loss, np.float32)
		"""
		#return gen_sen, sample_c

	def optimize(self,loss):
		optimizer = tf.train.AdamOptimizer(1e-3)
		gradients, variables = zip(*optimizer.compute_gradients(loss))
		gradients = [None if gradient is None else tf.clip_by_value(gradient,-1.0,1.0) for gradient in gradients]
		train_step = optimizer.apply_gradients(zip(gradients, variables))
		return train_step

	def sleepTrain(self, sess, enc_input, enc_len, labels):
		with tf.variable_scope(self.scope):
			dis_loss = trainer.discriminator.discrimi_loss(sess, enc_input, enc_len, labels)
			sess.run(self.optimizer(dis_loss))

