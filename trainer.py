import tensorflow as tf 
import Encoder
import Generator
import Discriminator

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

		self.encoder = Encoder(self.hid_units)
		self.generator = Generator(self.hid_units,self.batch_size,self.vocab_size)
		self.discriminator = Discriminator(self.c_size)

	def wakeTrain(self,sess,enc_input,enc_len,dec_input,dec_len,dec_tar):

		z = self.encoder.encode(sess,enc_input,enc_len)
		enc_outputs = self.encoder.output_logits(self,sess,enc_input,enc_len)
		reconst_loss = self.generator.reconst_loss(sess, enc_len, dec_len, z, enc_outputs, dec_tar, dec_input)
		logits, gen_sen, sample_c = self.generator.outputs(sess, enc_len, dec_len, z, enc_outputs, dec_tar, dec_input)
		
		logits = tf.reshape(logits, [-1, self.vocab_size])
		logit2word_embeds = tf.matmul(logits, self.word_embeds)
		logit_encode = tf.reshape(logit2word_embeds, [self.batch_size,-1,self.embed_size])

		c_loss = self.discriminator.discrimi_loss(logit_encode, dec_len, sample_c)
		z_loss = tf.mutual_information(z,new_z)

		generator_loss = reconst_loss + c_loss + z_loss
		sess.run(self.optimize(generator_loss))
		return gen_sen, sample_c

	def optimize(self,loss):
		optimizer = tf.train.AdamOptimizer(1e-3)
    	gradients, variables = zip(*optimizer.compute_gradients(loss))
    	gradients = [None if gradient is None else tf.clip_by_value(gradient,-1.0,1.0)
        				 for gradient in gradients]
    	self.train_step = optimizer.apply_gradients(zip(gradients, variables))
    	return self.train_step

    def sleepTrain(self, sess, enc_input, enc_len , labels):
    	code = self.encoder.encode(sess, enc_input, enc_len)
    	dis_loss = trainer.discriminator.discrimi_loss(code, labels)
    	sess.run(self.optimizer(dis_loss))

