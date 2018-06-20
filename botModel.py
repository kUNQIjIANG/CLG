import tensorflow as tf 

class botModel:
	def __init__(self,scope):
		self.scope = scope
		self.saver = tf.train.Saver()
	
	def save(self,sess,path):
		self.saver.save(sess,path)
		print("saved " + self.scope)

	def load(self,sess,path):
		self.saver.restore(sess,path)
		print("loaded " + self.scope)
		



