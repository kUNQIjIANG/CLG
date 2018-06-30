import tensorflow as tf 

class botModel:
	def __init__(self,scope):
		self.scope = scope
		self.saver = tf.train.Saver()
		self.save_path = './saved_oop/' + scope + '.ckpt'

	def save(self,sess):
		self.saver.save(sess,self.save_path)
		print("saved " + self.scope)

	def load(self,sess):
		self.saver.restore(sess,self.save_path)
		print("loaded " + self.scope)
		



