#!/usr/bin/env python3.5
import tensorflow as tf
import numpy as np
import gensim
import pickle
import os
from tensorflow.python.layers.core import Dense

#test push

#decoder_emb_inp = tf.get_variable("decoder_inp",[1,5,10],initializer = initializer)
tgt_sos_id = 0
tgt_eos_id = 1
batch_size = 50


if os.path.isfile("word_embeddings.pickle"):
    word_embeddings = pickle.load(open('word_embeddings.pickle','rb'))
    word2id = pickle.load(open('word2id.pickle','rb'))
    id2word = pickle.load(open('id2word.pickle','rb'))
    vocab_size = pickle.load(open('vocab_size.pickle','rb'))
    vocab = pickle.load(open('vocab.pickle','rb'))
else:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('glove.50d.txt', binary=False)
    vocab = []
    for joke in open("shorterjokes.txt").read().split("\n"):
        for word in joke.split():
            if word.lower() in word2vec.wv.vocab and word.lower() not in vocab:
                vocab.append(word.lower())
    print(len(vocab))

    vocab_size = len(vocab) + 3
    embed_size = 50
    word2id = {w:i+3 for i,w in enumerate(vocab)}
    id2word = {i+3: w for i,w in enumerate(vocab)}
    word2id["<Sos>"] = 0
    word2id["<Eos>"] = 1
    word2id["<Unk>"] = 2
    id2word[0] = "<Sos>"
    id2word[1] = "<Eos>"
    id2word[2] = "<Unk>"
    word_embeddings = np.zeros([vocab_size, embed_size])
    #Sos = np.zeros([1,embed_size])
    #Eos = np.ones([1,embed_size])
    #Unk = np.zeros([1,embed_size])

    for word in vocab:
        word_embeddings[word2id[word]] = word2vec[word]

    pickle.dump(word_embeddings, open("word_embeddings.pickle", "wb"))
    pickle.dump(word2id, open("word2id.pickle", "wb"))
    pickle.dump(id2word, open("id2word.pickle", "wb"))
    pickle.dump(vocab_size, open("vocab_size.pickle", "wb"))
    pickle.dump(vocab, open("vocab.pickle","wb"))


print("load word embedding done")


embed_size = 50
max_len = tf.placeholder(tf.int32, shape = [])
enc_sent = tf.placeholder(tf.int32, shape = [None, None])
dec_sent = tf.placeholder(tf.int32, shape = [None, None])
tar_sent = tf.placeholder(tf.int32, shape = [None, None])
source_len = tf.placeholder(tf.int32, shape = [None])

dec_seq_len = source_len + 1


hid_units = 50

initializer = tf.random_uniform_initializer(-0.1, 0.1)

word_embeddings = tf.cast(word_embeddings,tf.float32)
#tf.Variable(tf.random_uniform([vocab_size,embed_size],1,-1))

dec_embed = tf.nn.embedding_lookup(word_embeddings, dec_sent)
enc_embed = tf.nn.embedding_lookup(word_embeddings, enc_sent)

# encoder
shit_state = tf.contrib.rnn.LSTMStateTuple(tf.Variable(tf.random_normal([1,hid_units], mean=-1, stddev=4)),
                 						   tf.Variable(tf.random_normal([1,hid_units], mean=-1, stddev=4)))

encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hid_units)
encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, enc_embed, dtype = tf.float32,sequence_length = source_len)




latent_size = 50
u = tf.layers.dense(encoder_state.h,latent_size,tf.nn.sigmoid)
s = tf.layers.dense(encoder_state.h,latent_size,tf.nn.tanh)

z = u + s * tf.truncated_normal(tf.shape(u),1,-1)
dec_ini_state = tf.contrib.rnn.LSTMStateTuple(z,z)


z_list = []
for i in range(batch_size):
    b_z = tf.tile([z[i]],[max_len+1,1])
    z_list.append(b_z)

z_concat = tf.stack(z_list)
dec_input = tf.concat((dec_embed, z_concat),axis = 2)
"""
attention_states = tf.Variable(tf.random_uniform([1,10,hid_units],1,-1))
"""
attention_states = encoder_output
attention_mechanism = tf.contrib.seq2seq.LuongAttention(hid_units,
                                                        attention_states,
                                                        memory_sequence_length = source_len) # len of attention_states = source_seq_len

decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hid_units)
"""
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
												   initial_cell_state = dec_ini_state,
                                                   attention_layer_size = hid_units)
"""
# sequence_length is the decoder sequence lengh
# could be less than decoder input lengh but not more than decoder input lengh
#[10, 11, 10, 10, 10]
helper = tf.contrib.seq2seq.TrainingHelper(inputs = dec_input,sequence_length = dec_seq_len) 
                                           
greedyHelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(word_embeddings,tf.fill([batch_size],tgt_sos_id),tgt_eos_id)

initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size = batch_size)
# greedy helper is set up
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                          initial_state = initial_state,
                                          output_layer = Dense(vocab_size))

outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,maximum_iterations = None)
#

logits = outputs.rnn_output


hightest_indice = outputs.sample_id

seq_mask = tf.cast(tf.sequence_mask(dec_seq_len,max_len+1),tf.float32)
#for_con = tf.zeros([5,1,12144])
#logits = tf.concat([logits,for_con],axis = 1)
#logits = tf.reshape(logits,[5,11,12144])
#target = tf.reshape(tar_sent,[5,11])
seq_loss = tf.contrib.seq2seq.sequence_loss(logits, tar_sent, seq_mask,average_across_timesteps = False,average_across_batch = True)
kl_loss = 0.5 * (tf.reduce_sum(tf.square(s) + tf.square(u) - tf.log(tf.square(s))) - latent_size)
loss = tf.reduce_sum(seq_loss + kl_loss)

#train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
#optimizer = tf.train.AdamOptimizer(1e-3)
#gradients, variables = zip(*optimizer.compute_gradients(loss))
#gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
#train_step = optimizer.apply_gradients(zip(gradients, variables))

optimizer = tf.train.AdamOptimizer(1e-3)
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients = [
    None if gradient is None else tf.clip_by_value(gradient,-1.0,1.0)
    for gradient in gradients]
train_step = optimizer.apply_gradients(zip(gradients, variables))

#optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
#gvs = optimizer.compute_gradients(loss)
#clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
#train_step = optimizer.apply_gradients(clipped_gvs)

def next_batch(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    joke_data = open("shorterjokes.txt",'r').read().split('\n')
    generator = next_batch(joke_data,batch_size)

    batch_size = 50
    inp_max_len = 10
    #fall = [4,17,26,28,30,34,36,38,55,59,65,72,77]
    
    for step,jokes in enumerate(generator):
        inp_len = []
        for joke in jokes:
            inp_len.append(len(joke.split()))
        inp_max_len = max(inp_len)
        
        enc_inp = np.ones([batch_size,inp_max_len])
        for i,joke in enumerate(jokes):
            joke_word = joke.split()
            leng = len(joke_word)
            joke_w_id = []
            for word in joke_word:
                if word.lower() in vocab:
                    joke_w_id.append(word2id[word.lower()])
                else:
                    joke_w_id.append(word2id["<Unk>"])
            enc_inp[i,:leng] = joke_w_id
        

        sos_pad = np.zeros([batch_size,1])
        eos_pad = np.ones([batch_size,1])
        dec_outp = np.concatenate((enc_inp,eos_pad), axis = 1)
        dec_inp = np.concatenate((sos_pad,enc_inp), axis = 1)
        #print("step",step)
        #print("input len", inp_len)
        #print("enc_inp", enc_inp)
        #print("dec_inp",dec_inp)
        #print("dec_out",dec_outp)
        
        ind, l, outputs, _, z_c, d_i = sess.run([hightest_indice, loss, logits, train_step, z_concat, dec_input],
                                                    feed_dict = {enc_sent : enc_inp,
                                                                 dec_sent : dec_inp,
                                                                 tar_sent : dec_outp,
                                                                 source_len : inp_len,
                                                                 max_len: inp_max_len})
        if step % 1 == 0:
            print(l)
            #print(outputs.shape)
            #print("ind", ind)
            
            #print("seq_loss", s_l)
            #print("kl_loss", k_l)
            #print("loss", los)


            for sent in ind:
                print(' '.join([id2word[id] for id in sent]))
