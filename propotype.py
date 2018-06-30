#!/usr/bin/env python3.5
import tensorflow as tf
import numpy as np
import gensim
import pickle
import os
import random
from nltk import word_tokenize
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder

#test push

#decoder_emb_inp = tf.get_variable("decoder_inp",[1,5,10],initializer = initializer)
sos_id = 0
eos_id = 1
batch_size = 32
beam_width = 5



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
        for word in word_tokenize(joke):
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



"""
vari_enc_outputs = tf.zeros(tf.shape(encoder_output))
for i in range(batch_size):
    for j in range(2):
        u_output = tf.layers.dense(encoder_output[i,j,:],latent_size,tf.nn.sigmoid,name = "output_var_u")
        s_output = tf.layers.dense(encoder_output[i,j,:],latent_size,tf.nn.relu, name = "output_var_s")
        z_output = u_output + s_output * tf.truncated_normal(tf.shape(u_output),1,-1)
        vari_enc_outputs[i,j,:] = z_output
"""
"""
z_list = []
for i in range(batch_size):
    b_z = tf.tile([z[i]],[dec_max_len,1])
    z_list.append(b_z)

z_concat = tf.stack(z_list)
dec_input = tf.concat((dec_embed, z_concat),axis = 2)
"""

with tf.variable_scope("Training"):

    embed_size = 50
    
    dec_max_len = tf.placeholder(tf.int32, shape = [])
    enc_sent = tf.placeholder(tf.int32, shape = [None, None])
    dec_sent = tf.placeholder(tf.int32, shape = [None, None])
    tar_sent = tf.placeholder(tf.int32, shape = [None, None])
    enc_len = tf.placeholder(tf.int32, shape = [None])
    dec_len = tf.placeholder(tf.int32, shape = [None])
    schedule_kl_weight = tf.placeholder(tf.float32, shape = [])

    
    

    hid_units = 50

    initializer = tf.random_uniform_initializer(-0.1, 0.1)

    word_embeddings = tf.cast(word_embeddings,tf.float32)
    #tf.Variable(tf.random_uniform([vocab_size,embed_size],1,-1))

    dec_embed = tf.nn.embedding_lookup(word_embeddings, dec_sent)
    enc_embed = tf.nn.embedding_lookup(word_embeddings, enc_sent)

    # encoder
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hid_units)
    # a debug found : encoder output length depends on the max len 
    # in this batch, enc_len will be check, no matter emc_embed length(shape)
    encoder_output, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, enc_embed, dtype = tf.float32,sequence_length = enc_len)

    latent_size = 50

    u = tf.layers.dense(encoder_final_state.c,latent_size,tf.nn.sigmoid,name = "state_latent_u",reuse = tf.AUTO_REUSE)
    s = tf.layers.dense(encoder_final_state.c,latent_size,tf.nn.sigmoid,name = "state_latent_s",reuse = tf.AUTO_REUSE)

    z = u + s * tf.truncated_normal(tf.shape(u),1,-1)
    dec_ini_state = tf.contrib.rnn.LSTMStateTuple(z,z)

    dec_input = dec_embed

    # training and inference share this decoder cell
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hid_units, reuse = tf.AUTO_REUSE)

    vari_enc_outputs = encoder_output 


    train_attention_states = vari_enc_outputs
    train_attention_mechanism = tf.contrib.seq2seq.LuongAttention(hid_units,
                                                                  train_attention_states,
                                                                  memory_sequence_length = enc_len,
                                                                  ) # len of attention_states = source_seq_len

    train_atten_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, train_attention_mechanism,
    												   initial_cell_state = dec_ini_state,
                                                       attention_layer_size = hid_units)

    # sequence_length is the decoder sequence lengh
    # could be less than decoder input lengh but not more than decoder input lengh

    train_helper = tf.contrib.seq2seq.TrainingHelper(inputs = dec_input,sequence_length = dec_len) 
                                               

    # Since using attention cell, have to use this zero_state which is state.c for 
    # attention operation rather than a LSTMTuple(c,h)
    initial_state = train_atten_cell.zero_state(dtype=tf.float32, batch_size = batch_size)

    #initial_state = dec_ini_state
    train_decoder = tf.contrib.seq2seq.BasicDecoder(train_atten_cell, train_helper,
                                              initial_state = initial_state,
                                              output_layer = Dense(vocab_size))

    train_outputs, t_final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder,maximum_iterations = None)

    seq_mask = tf.cast(tf.sequence_mask(dec_len,dec_max_len),tf.float32)

    train_logits = train_outputs.rnn_output
    train_ind = train_outputs.sample_id

    seq_loss = tf.contrib.seq2seq.sequence_loss(train_logits, tar_sent, seq_mask,average_across_timesteps = False,average_across_batch = True)
    kl_loss = 0.5 * (tf.reduce_sum(tf.square(s) + tf.square(u) - tf.log(tf.square(s))) - latent_size)
    train_loss = seq_loss + schedule_kl_weight * kl_loss
    
    optimizer = tf.train.AdamOptimizer(1e-3)
    gradients, variables = zip(*optimizer.compute_gradients(train_loss))
    gradients = [
        None if gradient is None else tf.clip_by_value(gradient,-1.0,1.0)
        for gradient in gradients]
    train_step = optimizer.apply_gradients(zip(gradients, variables))

with tf.variable_scope("Training",reuse = True):

    beam_width = 5
    tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
        vari_enc_outputs, multiplier=beam_width)

    tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
        dec_ini_state, multiplier=beam_width)

    tiled_enc_len = tf.contrib.seq2seq.tile_batch(
        enc_len, multiplier=beam_width)

    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units=hid_units,
        memory=tiled_encoder_outputs,
        memory_sequence_length=tiled_enc_len)

    attention_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                         attention_layer_size = hid_units)

    decoder_initial_state = attention_cell.zero_state(dtype=tf.float32, batch_size= batch_size * beam_width)
    decoder_initial_state = decoder_initial_state.clone(cell_state=tiled_encoder_final_state)

    #tf.contrib.seq2seq.tile_batch(initial_state, beam_width),
    beam_decoder = beam_search_decoder.BeamSearchDecoder(cell=attention_cell,
                                                         embedding=word_embeddings,
                                                         start_tokens=tf.fill([batch_size],sos_id),
                                                         end_token=sos_id,
                                                         initial_state= decoder_initial_state,
                                                         beam_width=beam_width,
                                                         output_layer=Dense(vocab_size))

    
    greedyHelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(word_embeddings,tf.fill([batch_size],sos_id),eos_id)

    infer_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, greedyHelper,
                                              initial_state = dec_ini_state,
                                              output_layer = Dense(vocab_size))
    
    infer_outputs, i_final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(beam_decoder,maximum_iterations = dec_max_len)


    infer_ind = infer_outputs.predicted_ids[:,:,0]

#infer_ind = infer_outputs.sample_id

def next_batch_(ques, answ , batch_size):
    for i in range(0, len(ques) - batch_size, batch_size):
        yield ques[i:i+batch_size], answ[i:i+batch_size]

def next_batch(data, batch_size):
    for i in range(0, len(ques) - batch_size, batch_size):
        yield data[i:i+batch_size]

def max_batch_len(batch):
    batch_len = []
    
    for sent in batch:
        batch_len.append(len(word_tokenize(sent)))
        
    return batch_len, max(batch_len)

def batch_encode(batch, batch_size, inp_max_len, vocab, word2id, pid):
    enc_inp = pid * np.ones([batch_size,inp_max_len])
    for n, joke in enumerate(batch):
        joke_words = word_tokenize(joke)

        for iw, word in enumerate(joke_words):
            if word.lower() in vocab:
                enc_inp[n,iw] = word2id[word.lower()]
            else:
                enc_inp[n,iw] = (word2id["<Unk>"])
    return enc_inp

with tf.Session() as sess:
    saver = tf.train.Saver()
    model_path = './saved_with_scope/NML.ckpt'
    model_dir = 'saved_with_scope'

    if os.path.isdir(model_dir):
        print("Loading previous trained model ...")
        saver.restore(sess, model_path)
    else:
        sess.run(tf.global_variables_initializer())
        print("global initialing")
    
    joke_data = open("shorterjokes.txt",'r').read().split('\n')
    vocab.append("what's")
    epochs = 3

    for epoch in range(epochs):
        random.shuffle(joke_data)
        generator = next_batch(joke_data,batch_size)
        total_schedule = epochs * int(len(joke_data) / batch_size)
        
        for step, (quess,answs) in enumerate(generator):
        
            
            schedule =  epoch/epochs + step/total_schedule
            q_inp_len, q_inp_max_len = max_batch_len(quess)
            a_inp_len, a_inp_max_len = max_batch_len(answs)
            q_enc_inp = batch_encode(quess, batch_size, q_inp_max_len, vocab, word2id, 3)
            a_enc_inp = batch_encode(answs, batch_size, a_inp_max_len, vocab, word2id, 1)

            sos_pad = np.zeros([batch_size,1])
            eos_pad = np.ones([batch_size,1])
            dec_outp = np.concatenate((a_enc_inp,eos_pad), axis = 1)
            dec_inp = np.concatenate((sos_pad,a_enc_inp), axis = 1)
            
            # trianing graph
            t_ind, t_loss,  _ = sess.run([train_ind, train_loss, train_step],
                                                feed_dict = {enc_sent : q_enc_inp,
                                                             dec_sent : dec_inp,
                                                             tar_sent : dec_outp,
                                                             enc_len : q_inp_len,
                                                             dec_max_len : a_inp_max_len,
                                                             schedule_kl_weight : schedule})
            if step % 50 == 0:
                
                for tra, truth in zip(t_ind, dec_outp):
                    print("truth: " + ' '.join([id2word[id] for id in truth]))
                    print("tra: " + ' '.join([id2word[id] for id in tra]))
            
            
            # inference graph
            i_ind = sess.run(infer_ind,feed_dict = {enc_sent : q_enc_inp,
                                                     dec_sent : dec_inp,
                                                     tar_sent : dec_outp,
                                                     enc_len : q_inp_len,
                                                     dec_max_len : a_inp_max_len,
                                                     schedule_kl_weight : schedule})
            if step % 5 == 0:
                
                for inf, truth in zip(i_ind, dec_outp):
                    print("truth: " + ' '.join([id2word[id] for id in truth]))
                    print("inf: " + ' '.join([id2word[id] for id in inf]))                      
            
            

    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
