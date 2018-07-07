import tensorflow as tf 
import os
import trainer
import pickle
from trainer import Trainer
import random
import numpy as np
from nltk import word_tokenize


def next_batch(data, batch_size):
    for i in range(0, len(data) - batch_size, batch_size):
        yield data[i:i+batch_size]

def max_batch_len(batch):
    batch_len = []
    
    for sent in batch:
        batch_len.append(len(word_tokenize(sent)))
        
    return batch_len, max(batch_len)

def batch_encode(batch, batch_size, inp_max_len, vocab, word2id, pad_id):
    enc_inp = pad_id * np.ones([batch_size,inp_max_len])
    for n, joke in enumerate(batch):
        joke_words = word_tokenize(joke)

        for iw, word in enumerate(joke_words):
            if word.lower() in vocab:
                enc_inp[n,iw] = word2id[word.lower()]
            else:
                enc_inp[n,iw] = (word2id["<Unk>"])
    return enc_inp

hid_units = 100
batch_size = 32 
epochs = 5
c_size = 2
embed_size = 50
word_embeds = pickle.load(open('word_embeddings.pickle','rb'))
word2id = pickle.load(open('word2id.pickle','rb'))
id2word = pickle.load(open('id2word.pickle','rb'))
vocab_size = pickle.load(open('vocab_size.pickle','rb'))
vocab = pickle.load(open('vocab.pickle','rb'))

joke_data = open("shorterjokes.txt",'r').read().split('\n')

with tf.Session() as sess:

    word_embeds = tf.cast(word_embeds,tf.float32)

    trainer = Trainer(hid_units, batch_size, vocab_size, embed_size, c_size, word_embeds)
    
    if os.path.isdir('saved_oop'):
        trainer.encoder.load(sess)
        trainer.generator.load(sess)
        trainer.discriminator.load(sess)  
    else:
        sess.run(tf.global_variables_initializer())
        print("global initialing")

    for epoch in range(epochs):

        random.shuffle(joke_data)
        generator = next_batch(joke_data,batch_size)
        total_schedule = epochs * int(len(joke_data) / batch_size)
        
        for step, batch_joke in enumerate(generator):
            schedule =  epoch/epochs + step/total_schedule

            enc_label = np.random.rand(batch_size,2)

            inp_len, inp_max_len = max_batch_len(batch_joke)
            
            enc_inp = batch_encode(batch_joke, batch_size, inp_max_len, vocab, word2id, 3)
            

            sos_pad = np.zeros([batch_size,1])
            eos_pad = np.ones([batch_size,1])
            dec_outp = np.concatenate((enc_inp,eos_pad), axis = 1)
            dec_inp = np.concatenate((sos_pad,enc_inp), axis = 1)

            outp_len = inp_len + np.ones_like(inp_len)
            outp_max_len = inp_max_len + 1

            #
            gen_sen, gen_label = trainer.wakeTrain(sess, enc_inp, inp_len, dec_inp, outp_len, dec_outp)
            #print(gen_sen.shape)
            #print(enc_inp.shape)
            con_sen = np.concatenate((enc_inp, gen_sen[:,:-1]), axis = 0)
            con_lab = np.concatenate((enc_label, gen_label), axis = 0)
            con_len = np.concatenate((inp_len,inp_len), axis = 0)
            trainer.sleepTrain(sess, con_sen, con_len, con_lab)