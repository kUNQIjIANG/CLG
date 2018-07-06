import numpy as np 
import tensorflow as tf 
from encoder import Encoder
from generator import Generator
from discriminator import Discriminator

class Trainer:
	def __init__(self, hid_units, batch_size, vocab_size, embed_size, c_size, word_embeds):

		self.hid_units = hid_units
		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.c_size = c_size 
		self.word_embeds = word_embeds
		self.build_graph()
             
	def build_graph(self):

		self.encoder = Encoder(self.hid_units, self.word_embeds)
		self.generator = Generator(self.hid_units,self.batch_size,self.vocab_size,self.c_size, self.word_embeds, self.embed_size)
		self.discriminator = Discriminator(self.c_size,self.hid_units, self.batch_size, self.word_embeds, self.embed_size)

	def wakeTrain(self,sess,enc_input,enc_len,dec_input,dec_len,dec_tar):

		#enc_input = tf.cast(enc_input, tf.int32)
		#enc_embed = tf.nn.embedding_lookup(self.word_embeds, enc_input)
		
		z = self.encoder.encode(sess,enc_input,enc_len)
		enc_outputs = self.encoder.output_logits(sess,enc_input,enc_len)
		
		reconst_loss = self.generator.reconst_loss(sess, enc_len, dec_len, z, enc_outputs, dec_tar, dec_input)
		logits, gen_sen, sample_c = self.generator.outputs(sess, enc_len, dec_len, z, enc_outputs, dec_tar, dec_input)
		
		logits = tf.reshape(logits, [-1, self.vocab_size])
		logit2word_embeds = tf.matmul(logits, self.word_embeds)
		logit_encode = tf.reshape(logit2word_embeds, [self.batch_size,-1,self.embed_size])

		logit_encode = sess.run(logit_encode)
		
		#sample_c = sess.run(sample_c)
		# length of generated sentence ?
		c_loss = self.discriminator.discrimi_loss(sess, logit_encode, dec_len, sample_c)

		# 1, force encode focus on unstructure
		# 2, ensure unstructure part of generated sentence unchaged
		new_z = self.encoder.encode(sess, gen_sen, dec_len)
		z_loss = tf.nn.l2_loss(z.c - new_z.c)

		c_loss = tf.convert_to_tensor(reconst_loss, np.float32)

		generator_loss = c_loss 
		sess.run(self.optimize(generator_loss))
		return gen_sen, sample_c

	def optimize(self,loss):
		optimizer = tf.train.AdamOptimizer(1e-3)
		gradients, variables = zip(*optimizer.compute_gradients(loss))
		gradients = [None if gradient is None else tf.clip_by_value(gradient,-1.0,1.0) for gradient in gradients]
		self.train_step = optimizer.apply_gradients(zip(gradients, variables))
		return self.train_step

	def sleepTrain(self, sess, enc_input, enc_len, labels):
		
		dis_loss = trainer.discriminator.discrimi_loss(sess, enc_input, enc_len, labels)
		sess.run(self.optimizer(dis_loss))

