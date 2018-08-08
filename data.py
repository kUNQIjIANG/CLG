import tensorflow as tf 
import numpy as np 
from configs import args

class DataFlow:
	def __init__(self):
		index_from = 4
		self.word2id = tf.keras.datasets.imdb.get_word_index()
		self.word2id = {k : v+index_from for k, v in self.word2id.items() if v<args.vocab_size-index_from}
		self.word2id['<pad>'] = 0
		self.word2id['<sos>'] = 1
		self.word2id['<unk>'] = 2
		self.word2id['<eos>'] = 3
		self.word2id['<spc>'] = 4 
		self.id2word = {k : w for w,k in self.word2id.items()}
		self.vocab_size = args.vocab_size 
		self.max_len = args.max_len
		self.batch_size = args.batch_size

	def word_dropout(self,x):
		is_dropout = np.random.binomial(1, args.word_dropout_rate, x.shape)
		apply_dropout = np.vectorize(lambda x, k: self.word2id['<unk>'] if (k and (x not in range(4))) else x)
		return apply_dropout(x, is_dropout)

	def load(self):
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words = self.vocab_size, index_from = 4)
		x = np.concatenate((x_train,x_test))
		y = np.concatenate((y_train,y_test))

		post_x = tf.keras.preprocessing.sequence.pad_sequences(x, self.max_len + 1,
													truncating = 'post', padding = 'post')
		pre_x = tf.keras.preprocessing.sequence.pad_sequences(x, self.max_len + 1,
													truncating = 'pre', padding = 'post')

		x = np.concatenate((pre_x,post_x))
		y = np.concatenate((y,y))
		
		enc_inp = x[:, 1:]
		dec_inp = self.word_dropout(x)
		dec_tar = np.concatenate((x[:,1:], np.full([x.shape[0],1],self.word2id['<eos>'])), axis = -1)
		
		dataPipe = tf.data.Dataset.from_tensor_slices((enc_inp,dec_inp,dec_tar,y))
		dataPipe = dataPipe.shuffle(len(enc_inp)).batch(self.batch_size)
		iterator = dataPipe.make_initializable_iterator()

		return iterator, len(enc_inp)

