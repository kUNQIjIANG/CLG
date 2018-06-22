import tensorflow as tf 
import Encoder
import Generator
import Discriminator

class Train:
	def __init__(self, hid_units, batch_size, vocab_size, c_size):

		self.hid_units = hid_units
		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.c_size = c_size 

	def build_graph(self):

		encoder = Encoder(self.hid_units)
		generator = Generator(self.hid_units,self.batch_size,self.vocab_size)
		discriminator = Discriminator(self.c_size)

	def train_generator(self,sess,enc_input,enc_len,dec_input,dec_len,dec_tar):

		z = encoder.encode(sess,enc_input,enc_len)
		enc_outputs = encoder.output_logits(self,sess,enc_input,enc_len)
		reconst_loss = generator.reconst_loss(sess, enc_len, dec_len, z, enc_outputs, dec_tar, dec_input)
		logits = generator.output_logits(sess, enc_len, dec_len, z, enc_outputs, dec_tar, dec_input)
		new_z = encoder.encode(sess,logits,dec_len)
		predict_c = discriminator.predict_c(sess,new_z)
		c_loss = tf.cross_entropy_loss_with_logits(label = true_c, logits = predict_c)
		z_loss = tf.mutual_information(z,new_z)

		generator_loss = reconst_loss + c_loss + z_loss

		optimizer = tf.train.AdamOptimizer(1e-3)
    	gradients, variables = zip(*optimizer.compute_gradients(generator_loss))
    	gradients = [None if gradient is None else tf.clip_by_value(gradient,-1.0,1.0)
        				 for gradient in gradients]
    	self.train_step = optimizer.apply_gradients(zip(gradients, variables))
    	sess.run(self.train_step)
