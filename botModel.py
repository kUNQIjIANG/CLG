import tensorflow as tf 

class botModel:
	def __init__(self,scope):
		self.scope = scope
		
		self.save_path = './saved_oop/' + scope + '.ckpt'

	def save(self,sess):
		tf.train.Saver().save(sess,self.save_path)
		print("saved " + self.scope)

	def load(self,sess):
		tf.train.Saver().restore(sess,self.save_path)
		print("loaded " + self.scope)
		



