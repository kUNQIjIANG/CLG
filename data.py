import tensorflow as tf 
import numpy as np 

class DataFlow:
	def __init__(self, vocab_size, max_len, batch_size):
		index_from = 4
		self.word2id = tf.keras.datasets.imdb.get_word_index()
		self.word2id = {k : v+index_from for k, v in self.word2id.items() if v<vocab_size-index_from}
		self.word2id['<pad>'] = 0
		self.word2id['<sos>'] = 1
		self.word2id['<unk>'] = 2
		self.word2id['<eos>'] = 3
		self.word2id['<spc>'] = 4 
		self.id2word = {k : w for w,k in self.word2id.items()}
		self.vocab_size = vocab_size 
		self.max_len = max_len
		self.batch_size = batch_size

	def load(self):
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words = self.vocab_size, index_from = 4)
		data = np.concatenate((x_train,x_test))

		pre_trunc = tf.keras.preprocessing.sequence.pad_sequences(data, self.max_len + 1,
													truncating = 'pre', padding = 'post')
		post_trunc = tf.keras.preprocessing.sequence.pad_sequences(data, self.max_len + 1,
													truncating = 'post', padding = 'post')
		aug_data = np.concatenate((pre_trunc, post_trunc))

		enc_inp = aug_data[:, 1:]
		dec_inp = aug_data
		dec_tar = np.concatenate((aug_data[:,1:], np.full([aug_data.shape[0],1],1)), axis = -1)

		dataPipe = tf.data.Dataset.from_tensor_slices((enc_inp,dec_inp,dec_tar))
		dataPipe = dataPipe.shuffle(len(enc_inp)).batch(self.batch_size)
		iterator = dataPipe.make_initializable_iterator()

		return iterator 

