#!/usr/bin/env python3.5
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense

#test push

#decoder_emb_inp = tf.get_variable("decoder_inp",[1,5,10],initializer = initializer)
tgt_sos_id = 0
tgt_eos_id = 1
batch_size = 5
"""
l = open("word.txt").read().split()
vocab_size = len(l)
print(vocab_size)
print(l)
"""

wordVec = open("glove.50d.txt").read().split('\n')
vocab_size = int(wordVec[0].split()[0]) + 2
embed_size = int(wordVec[0].split()[1])

SOS = np.random.randn(1,embed_size)
EOS = np.random.randn(1,embed_size)
word2id = {"<SOS>":0,"<EOS>":1}
id2word = {0:"<SOS>",1:"<EOS>"}
word_embeds = np.zeros([vocab_size,embed_size])
word_embeds[0] = SOS
word_embeds[1] = EOS
for i, word_line in enumerate(wordVec[1:-1]):
    line_split = word_line.split()
    word = line_split[0]
    vec = line_split[1:]
    word2id[word] = i+2
    id2word[i+2] = word
    word_embeds[i+2] = list(map(float,vec))
word2id["what's"] = word2id["what"]
id2word[word2id["what"]] = "what's"

print("load word embedding done")

#source_len = 7
embed_size = 50
max_len = tf.placeholder(tf.int32, shape = [])
dec_sent = tf.placeholder(tf.int32, shape = [None, None])
enc_sent = tf.placeholder(tf.int32, shape = [None, None])
source_len = tf.placeholder(tf.int32, shape = [None])


hid_units = 5

initializer = tf.random_uniform_initializer(-0.1, 0.1)

word_embeddings = tf.cast(word_embeds,tf.float32)
#tf.Variable(tf.random_uniform([vocab_size,embed_size],1,-1))

dec_embed = tf.nn.embedding_lookup(word_embeddings, dec_sent)
enc_embed = tf.nn.embedding_lookup(word_embeddings, enc_sent)

# encoder
shit_state = tf.contrib.rnn.LSTMStateTuple(tf.Variable(tf.random_normal([1,hid_units], mean=-1, stddev=4)),
                 						   tf.Variable(tf.random_normal([1,hid_units], mean=-1, stddev=4)))

encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hid_units)
encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, enc_embed, dtype = tf.float32,sequence_length = source_len)




latent_size = 5
u = tf.layers.dense(encoder_state.h,5,tf.nn.sigmoid)
s = tf.layers.dense(encoder_state.h,5,tf.nn.tanh)

z = u + s * tf.truncated_normal(tf.shape(u),1,-1)
z = tf.contrib.rnn.LSTMStateTuple(z,z)



"""
attention_states = tf.Variable(tf.random_uniform([1,10,hid_units],1,-1))
"""
attention_states = encoder_output
attention_mechanism = tf.contrib.seq2seq.LuongAttention(hid_units,
                                                        attention_states,
                                                        memory_sequence_length = source_len) # len of attention_states = source_seq_len

decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hid_units)

decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
												   initial_cell_state = z,
                                                   attention_layer_size = hid_units)

helper = tf.contrib.seq2seq.TrainingHelper(inputs = dec_embed,
                                           sequence_length = [6]) # sequence_length is the decoder sequence lengh
                                                                  # could be less than decoder input lengh but not more than decoder input lengh
greedyHelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(word_embeddings,tf.fill([batch_size],tgt_sos_id),tgt_eos_id)

initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size = batch_size)
# greedy helper is set up
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, greedyHelper,
                                          initial_state = initial_state,
                                          output_layer = Dense(vocab_size))

outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,maximum_iterations = 10)
#

logits = outputs.rnn_output
print(logits.shape)
hightest_indice = outputs.sample_id

seq_mask = tf.cast(tf.sequence_mask(source_len,max_len),tf.float32)
#target = tf.reshape(enc_sent,[1,7])
#logits = tf.reshape(logits, [1,7,57])
target = enc_sent
seq_loss = tf.contrib.seq2seq.sequence_loss(logits, target, seq_mask,average_across_timesteps = False,average_across_batch = True)
kl_loss = 0.5 * tf.reduce_sum(tf.exp(s) + tf.square(u) - 1 - s)
loss = seq_loss + kl_loss

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
#gvs = optimizer.compute_gradients(loss)
#clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
#train_step = optimizer.apply_gradients(clipped_gvs)
def next_batch(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

with tf.Session() as sess:

    """
    word2id = {w:i for i,w in enumerate(l)}
    id2word = {i:w for i,w in enumerate(l)}


    encoder_sent = ["we presume the scale is pages all",
                    "every month Google re-indexing! web pages scale",
                    "all pages of the web"]

    encoder_word = [sent.split() for sent in encoder_sent]
    encoder_seq_len = [len(s) for s in encoder_word]
    enc_max_len = np.max(encoder_seq_len)
    enc_sent_id = [[word2id[w] for w in words] for words in encoder_word]
    enc_id_matrix = np.full([len(encoder_sent),enc_max_len], 0)
    for i, sent_id in enumerate(enc_sent_id):
        enc_id_matrix[i,:len(sent_id)] = sent_id

    sent = "the web Google does presume logarithmic month If presume maximum PR the we scale"
    sent_word = sent.split()
    sent_id = [word2id[w] for w in sent_word]
    test_sent = np.array([sent_id])
    """


    joke_data = open("shorterjokes.txt",'r').read().split('\n')

    generator = next_batch(joke_data,5)

    batch_size = 5
    inp_max_len = 10
    enc_inp = np.ones([batch_size,inp_max_len])
    for _,jokes in enumerate(generator):
        inp_len = []
        for i,joke in enumerate(jokes):
            joke_word = joke.split()
            leng = len(joke_word)
            inp_len.append(leng)
            joke_w_id = []
            for word in joke_word:
                if word.lower() in word2id.keys():
                    joke_w_id.append(word2id[word.lower()])
                else:
                    joke_w_id.append(word2id["<unk>"])
            enc_inp[i,:leng] = joke_w_id
        break

    sos_pad = np.zeros([5,1])
    dec_inp = np.concatenate((sos_pad,enc_inp), axis = 1)
    sess.run(tf.global_variables_initializer())
    outputs, ind, s_l, k_l, los, _ = sess.run([logits,hightest_indice,seq_loss,kl_loss,loss, optimizer],
                                                feed_dict = {enc_sent : enc_inp,
                                                             dec_sent : dec_inp,
                                                             source_len : inp_len,
                                                             max_len: inp_max_len})


    print(outputs.shape)
    print("ind", ind)
    print("seq_loss", s_l)
    print("kl_loss", k_l)
    print("loss", los)


    for sent in ind:
        print(' '.join([id2word[id] for id in sent]))
