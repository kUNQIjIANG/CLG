import tensorflow as tf
import botModel

class Discriminator(botModel):
	def __init__(self,c_size):
		self.super.__init__('discriminator')
		self.c_size = c_size
		
		self.build_graph()
		
	def build_graph(self):
		with tf.variable_scope(self.scope):
			self.enc_state = tf.placeholders(tf.float32, shape = [None, None])
			self.proj_c = tf.layers.dense(self.enc_state, self.c_size)

	def predict_c(self,sess,enc_state):
		return sess.run(self.proj_c,{self.enc_state : enc_state})
