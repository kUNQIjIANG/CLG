#!/usr/bin/env python3.5
import tensorflow as tf
import numpy as np
import gensim
import pickle
import os
import random
from tensorflow.python.layers.core import Dense

#test push

#decoder_emb_inp = tf.get_variable("decoder_inp",[1,5,10],initializer = initializer)
sos_id = 0
eos_id = 1
batch_size = 50


if os.path.isfile("word_embeddings.pickle"):
    word_embeddings = pickle.load(open('word_embeddings.pickle','rb'))
    word2id = pickle.load(open('word2id.pickle','rb'))
    id2word = pickle.load(open('id2word.pickle','rb'))
    vocab_size = pickle.load(open('vocab_size.pickle','rb'))
    vocab = pickle.load(open('vocab.pickle','rb'))
    print("loading done")
else:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('glove.50d.txt', binary=False)
    vocab = []
    for joke in open("shorterjokes.txt").read().split("\n"):
        for word in joke.split():
            if word.lower() in word2vec.wv.vocab and word.lower() not in vocab:
                vocab.append(word.lower())
    print(len(vocab))

    vocab_size = len(vocab) + 4
    embed_size = 50
    word2id = {w:i+4 for i,w in enumerate(vocab)}
    id2word = {i+4: w for i,w in enumerate(vocab)}
    word2id["<Sos>"] = 0
    word2id["<Eos>"] = 1
    word2id["<Unk>"] = 2
    word2id["<Pad>"] = 3
    id2word[0] = "<Sos>"
    id2word[1] = "<Eos>"
    id2word[2] = "<Unk>"
    id2word[3] = "<Pad>"
    word_embeddings = np.random.rand(vocab_size, embed_size)
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

    print("construct word embedding done")


embed_size = 50
max_len = tf.placeholder(tf.int32, shape = [])
enc_sent = tf.placeholder(tf.int32, shape = [None, None])
dec_sent = tf.placeholder(tf.int32, shape = [None, None])
tar_sent = tf.placeholder(tf.int32, shape = [None, None])
source_len = tf.placeholder(tf.int32, shape = [None])
#batch_size = tf.placeholder(tf.int32, shape = [])

dec_seq_len = source_len + 1
dec_max_len = max_len + 1

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

"""
z_list = []
for i in range(batch_size):
    b_z = tf.tile([z[i]],[dec_max_len,1])
    z_list.append(b_z)

z_concat = tf.stack(z_list)
dec_input = tf.concat((dec_embed, z_concat),axis = 2)
"""
dec_input = dec_embed

attention_states = encoder_output
attention_mechanism = tf.contrib.seq2seq.LuongAttention(hid_units,
                                                        attention_states,
                                                        memory_sequence_length = source_len) # len of attention_states = source_seq_len
# training and inference share this decoder cell
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hid_units)

decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
												   initial_cell_state = dec_ini_state,
                                                   attention_layer_size = hid_units)

# sequence_length is the decoder sequence lengh
# could be less than decoder input lengh but not more than decoder input lengh

train_helper = tf.contrib.seq2seq.TrainingHelper(inputs = dec_input,sequence_length = dec_seq_len) 
                                           
greedyHelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(word_embeddings,tf.fill([batch_size],sos_id),eos_id)

initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size = batch_size)

train_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, train_helper,
                                          initial_state = initial_state,
                                          output_layer = Dense(vocab_size))

infer_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, greedyHelper,
                                          initial_state = initial_state,
                                          output_layer = Dense(vocab_size))

train_outputs, t_final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder,maximum_iterations = None)

infer_outputs, i_final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(infer_decoder,maximum_iterations = dec_max_len)

train_ind = train_outputs.sample_id
infer_ind = infer_outputs.sample_id

train_logits = train_outputs.rnn_output

seq_mask = tf.cast(tf.sequence_mask(dec_seq_len,dec_max_len),tf.float32)

seq_loss = tf.contrib.seq2seq.sequence_loss(train_logits, tar_sent, seq_mask,average_across_timesteps = False,average_across_batch = True)
kl_loss = 0.5 * (tf.reduce_sum(tf.square(s) + tf.square(u) - tf.log(tf.square(s))) - latent_size)
#loss = tf.reduce_sum(seq_loss + 0.1 * kl_loss)
#loss = kl_loss 
train_loss = seq_loss


optimizer = tf.train.AdamOptimizer(1e-3)
gradients, variables = zip(*optimizer.compute_gradients(train_loss))
gradients = [
    None if gradient is None else tf.clip_by_value(gradient,-1.0,1.0)
    for gradient in gradients]
train_step = optimizer.apply_gradients(zip(gradients, variables))


def next_batch(data, batch_size):
    for i in range(0, len(data) - batch_size, batch_size):
        yield data[i:i+batch_size]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    joke_data = open("shorterjokes.txt",'r').read().split('\n')
    vocab.append("what's")
    epochs = 5
    for epoch in range(epochs):
        random.shuffle(joke_data)
        generator = next_batch(joke_data,batch_size)
        
        for step,jokes in enumerate(generator):
            inp_len = []
            batch_rec = []
            for joke in jokes:
                word_in_voc = [word for word in joke.split() if word in vocab]
                if len(word_in_voc) == 0:
                    word_in_voc = ["I","am","here"]
                inp_len.append(len(word_in_voc))
                joke_w_id = []
                for word in word_in_voc:
                    if word.lower() == "what's":
                        joke_w_id.append(word2id["what"])
                    else:
                        joke_w_id.append(word2id[word.lower()])
                batch_rec.append(joke_w_id)

            inp_max_len = max(inp_len)

            enc_inp = 3 * np.ones([batch_size,inp_max_len])
            for i, (w_id,leng) in enumerate(zip(batch_rec, inp_len)):
                #print("w_id", w_id)
                #print("len", leng)
                enc_inp[i,:leng] = w_id
                
            sos_pad = np.zeros([batch_size,1])
            pad_pad = 3 * np.ones([batch_size,1])
            dec_outp = np.concatenate((enc_inp,pad_pad), axis = 1)
            dec_inp = np.concatenate((sos_pad,enc_inp), axis = 1)
            for i, leng in enumerate(inp_len):
                dec_outp[i,leng] = word2id["<Eos>"]
            

            #print("step",step)
            #print("input len", inp_len)
            #print("enc_inp", enc_inp)
            #print("dec_inp",dec_inp)
            #print("dec_out",dec_outp)
            
            t_ind, i_ind, t_loss,  _ = sess.run([train_ind, infer_ind, train_loss, train_step],
                                                        feed_dict = {enc_sent : enc_inp,
                                                                     dec_sent : dec_inp,
                                                                     tar_sent : dec_outp,
                                                                     source_len : inp_len,
                                                                     max_len: inp_max_len})
            if step % 50 == 0:
                
                for tra, inf, truth in zip(t_ind, i_ind, dec_outp):
                    print("truth: " + ' '.join([id2word[id] for id in truth]))
                    print("tra: " + ' '.join([id2word[id] for id in tra]))
                    print("inf: " + ' '.join([id2word[id] for id in inf]))
    
    saver = tf.train.Saver()
    model_path = './saved/NML.ckpt'
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
