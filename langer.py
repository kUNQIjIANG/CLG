import tensorflow as tf 
import os
import trainer
import pickle
from trainer import Trainer
import random
import numpy as np
from nltk import word_tokenize
from data import DataFlow 
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder

hid_units = 100
batch_size = 32 
epochs = 1
c_size = 2
beam_width = 5
embed_size = 300
sos_id = 0
eos_id = 1
vocab_size = 20000 
max_len = 15
latent_size = 100

with tf.variable_scope("Training"):

    
    dec_max_len = tf.placeholder(tf.int32, shape = [],name = '1')
    enc_sent = tf.placeholder(tf.int32, shape = [None, None], name  = '2')
    dec_sent = tf.placeholder(tf.int32, shape = [None, None], name ='3')
    tar_sent = tf.placeholder(tf.int32, shape = [None, None], name = '4')
    enc_len = tf.placeholder(tf.int32, shape = [None], name = '5')
    dec_len = tf.placeholder(tf.int32, shape = [None], name = '66')
    schedule_kl_weight = tf.placeholder(tf.float32, shape = [], name = '7')

    
    
    word_embeddings = tf.get_variable("word_embeds",[vocab_size, embed_size])
    

    initializer = tf.random_uniform_initializer(-0.1, 0.1)

    #tf.Variable(tf.random_uniform([vocab_size,embed_size],1,-1))

    dec_embed = tf.nn.embedding_lookup(word_embeddings, dec_sent)
    enc_embed = tf.nn.embedding_lookup(word_embeddings, enc_sent)

    # encoder
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hid_units)
    # a debug found : encoder output length depends on the max len 
    # in this batch, enc_len will be check, no matter emc_embed length(shape)
    encoder_output, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, enc_embed, dtype = tf.float32,sequence_length = enc_len)

    

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

    seq_loss = tf.contrib.seq2seq.sequence_loss(train_logits, tar_sent, seq_mask,
                    average_across_timesteps = False,average_across_batch = True)
    
    kl_loss = 0.5 * tf.reduce_sum(tf.square(s) + tf.square(u) - tf.log(tf.square(s)) - 1) / batch_size
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



step = 1
with tf.Session() as sess:
    saver = tf.train.Saver()
    model_path = './saved_with_imdb/NML.ckpt'
    model_dir = 'saved_with_imdb'

    if os.path.isdir(model_dir):
        print("Loading previous trained model ...")
        saver.restore(sess, model_path)
    else:
        sess.run(tf.global_variables_initializer())
        
        print("global initialing")
    
    data = DataFlow(vocab_size, max_len, batch_size)
    iterator, total_len = data.load()
    total_schedule = epochs * int(total_len / batch_size)

    for epoch in range(epochs):

        sess.run(iterator.initializer)
        while True:
            try:
                enc_inp, dec_inp, dec_tar, enc_label = sess.run(iterator.get_next())
                
                enc_length, dec_length = sess.run([tf.count_nonzero(enc_inp, axis = 1),
                                             tf.count_nonzero(dec_inp, axis = 1)])
                
                enc_label = sess.run(tf.one_hot(enc_label, depth = 2))       

                schedule =  step/total_schedule
                
            # trianing graph
                
                t_ind, _, k_loss, s_loss = sess.run([train_ind, train_step, kl_loss, seq_loss],
                                                    feed_dict = {enc_sent : enc_inp,
                                                                 dec_sent : dec_inp,
                                                                 tar_sent : dec_tar,
                                                                 enc_len : enc_length,
                                                                 dec_len : dec_length,
                                                                 dec_max_len : max(dec_length),
                                                                 schedule_kl_weight : schedule})
                
                if step % 10 == 0:
                    print("step: {}, kl_w: {}, kl_l: {}".format(step, schedule, k_loss))
                    print("step: {} seq_loss: {}".format(step,s_loss))
                    for tra, truth in zip(t_ind, dec_tar):
                        print("tru: " + ' '.join([data.id2word[id] for id in truth]))
                        print("tra: " + ' '.join([data.id2word[id] for id in tra]))
                
                
                step += 1
                # inference graph
                """
                i_ind = sess.run(infer_ind,feed_dict = {enc_sent : q_enc_inp,
                                                         dec_sent : dec_inp,
                                                         tar_sent : dec_outp,
                                                         enc_len : q_inp_len,
                                                         dec_max_len : q_inp_max_len+1,
                                                         schedule_kl_weight : schedule})
                if step % 5 == 0:
                    
                    for inf, truth in zip(i_ind, dec_tar):
                        print("truth: " + ' '.join([id2word[id] for id in truth]))
                        print("inf: " + ' '.join([id2word[id] for id in inf]))                      
                
                """
            except tf.errors.OutOfRangeError:
                    break

    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)
