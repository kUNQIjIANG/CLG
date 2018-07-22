import tensorflow as tf 
import os
import trainer
import pickle
from trainer import Trainer
import random
import numpy as np
from nltk import word_tokenize
from data import DataFlow 

hid_units = 100
batch_size = 32 
epochs = 2
c_size = 2
beam_width = 5
embed_size = 300
sos_id = 0
eos_id = 1
vocab_size = 20000
max_len = 15
save_path = './d_saved/'

test = ['this is one of the best movies i think',
        'i never saw a image as terrible as this',
        'first time to see such a great movie',
        'the boring film is a problem for me']

with tf.Session() as sess:
    word_embeds = tf.get_variable("word_embeds",[vocab_size, embed_size])

    trainer = Trainer(hid_units, batch_size, vocab_size, embed_size, c_size, word_embeds, sos_id, eos_id,beam_width)
    data = DataFlow(vocab_size, max_len, batch_size)
    iterator, total_len = data.load()
    next_element = iterator.get_next()
    total_step = epochs * total_len / batch_size

    test_len = []
    for sen in test:
        test_len.append(len(sen.split()))
    test_inp = np.zeros((len(test),max(test_len)))
    for i,sen in enumerate(test):
        for j,word in enumerate(sen.split()):
            test_inp[i,j] = data.word2id[word]

    given_c = np.array([[1,0],
                        [0,1],
                        [1,0],
                        [0,1]])
    
    if os.path.isdir('d_saved'):
        print("loading model from {}".format(save_path))
        trainer.saver.restore(sess,save_path)
        #trainer.encoder.load(sess)
        #trainer.generator.load(sess)
        #trainer.discriminator.load(sess)  
    else:
        sess.run(tf.global_variables_initializer())
        print("global initialized")
    step = 0
    for epoch in range(epochs):
        print("epoch---------{}".format(epoch))
        sess.run(iterator.initializer)
        while True:
            try:
                enc_inp, dec_inp, dec_tar, enc_label = sess.run(next_element)
                
                enc_len, dec_len = sess.run([tf.count_nonzero(enc_inp, axis = 1),
                							 tf.count_nonzero(dec_inp, axis = 1)])
                
                enc_label = sess.run(tf.one_hot(enc_label, depth = 2))
            
                # for inference desired c
                # given_c = np.concatenate((np.zeros([enc_inp.shape[0],1]),np.ones([enc_inp.shape[0],1])),axis=1)
                
                # pre-trian discriminator with supervised label
                kl_weight = step / total_step
                if step < 2000:
                    vae_loss, vae_rec, vae_kl, vae_sen, vae_u, vae_s = trainer.vaeTrain(sess, enc_inp, enc_len, dec_inp, dec_len, dec_tar,kl_weight)
                    pre_loss, pre_discri_acc = trainer.preTrain(sess, enc_inp, enc_len, enc_label)

                    if step % 20 == 0:

                        inf_ids = trainer.inference(sess,test_inp, test_len, given_c)
                        for tr, truth in zip(inf_ids, test_inp):
                            print("step: {} ".format(step) + "tru: " + ' '.join([data.id2word[id] for id in truth]))
                            print("step: {} ".format(step) + "inf: " + ' '.join([data.id2word[id] for id in tr]))         

                        print("epoch lenght: {}".format(total_len/batch_size))
                        print("step: {}, kl_w: {}, vae_u: {}, vae_s: {}".format(step,kl_weight,vae_u,vae_s))
                        print("step: {}, vae_loss : {}, vae_kl : {}, vae_rec: {}".format(step,vae_loss, vae_kl, vae_rec))
                        print("step: {}, pre-train loss : {}, accuracy : {}".format(step,pre_loss, pre_discri_acc))
                        for tr, truth in zip(vae_sen, dec_tar):
                            print("step: {} ".format(step) + "tru: " + ' '.join([data.id2word[id] for id in truth]))
                            print("step: {} ".format(step) + "vae: " + ' '.join([data.id2word[id] for id in tr]))          
                else:
                    # wake phase
                    gen_sen, gen_label, c_loss,z_loss,kl_loss, rec_loss, syn_acc, mean, sig = trainer.wakeTrain(sess, enc_inp, enc_len, dec_inp, dec_len, dec_tar,kl_weight)
                    #con_sen = np.concatenate((enc_inp, gen_sen[:,:-1]), axis = 0)
                    #con_lab = np.concatenate((enc_label, gen_label), axis = 0)
                    #con_len = np.concatenate((enc_len,enc_len), axis = 0)
                    
                    # sleep phase
                    sleep_loss, sleep_acc = trainer.sleepTrain(sess, enc_inp, enc_len, enc_label, gen_sen, dec_len, gen_label)
                    

                    # trianing output
                
                    if step % 20 == 0:
                        
                        inf_ids = trainer.inference(sess,test_inp, test_len, given_c)
                        for tr, truth in zip(inf_ids, test_inp):
                            print("step: {} ".format(step) + "tru: " + ' '.join([data.id2word[id] for id in truth]))
                            print("step: {} ".format(step) + "inf: " + ' '.join([data.id2word[id] for id in tr]))         
                        
                        print("step: {}, mean: {}, sig: {}".format(step, mean, sig))
                        print("step: {}, kl_weight: {}, c_loss: {}, z_loss: {}".format(step,kl_weight,c_loss,z_loss))
                        print("step: {}, kl_loss: {} rec_loss: {}, syn_acc: {}".format(step,kl_loss,rec_loss, syn_acc))
                        print("step: {}, sleep loss : {}, accuracy : {}".format(step, sleep_loss, sleep_acc))
                        for tr, truth in zip(gen_sen, dec_tar):
                            print("step: {} ".format(step) + "tru: " + ' '.join([data.id2word[id] for id in truth]))
                            print("step: {} ".format(step) + "tra: " + ' '.join([data.id2word[id] for id in tr])) 
                    
                    """
                    # inference output 
                    if step % 1 == 0:
                        for inf, truth in zip(inf_ids, dec_tar):
                            print("truth: " + ' '.join([data.id2word[id] for id in truth]))
                            print("inf: " + ' '.join([data.id2word[id] for id in inf])) 
                    
                    """
                step += 1
            except tf.errors.OutOfRangeError:
                save_path = trainer.saver.save(sess, save_path)
                print("Model saved in file: %s" % save_path)
                break
    

