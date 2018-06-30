import tensorflow as tf 
import os
import trainer

hid_units = 100
batch_size = 32 
vocab_size = 10000
c_size = 1

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

	trainer = Trainer(hid_units, batch_size, vocab_size, c_size)
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

        	enc_label = np.ones((1,batch_size))

            inp_len, inp_max_len = max_batch_len(quess)
            
            enc_inp = batch_encode(quess, batch_size, q_inp_max_len, vocab, word2id, 3)
            

            sos_pad = np.zeros([batch_size,1])
            eos_pad = np.ones([batch_size,1])
            dec_outp = np.concatenate((enc_inp,eos_pad), axis = 1)
            dec_inp = np.concatenate((sos_pad,enc_inp), axis = 1)

            outp_len = inp_len + np.ones_like(inp_len)
            outp_max_len = inp_max_len + 1

            gen_sen, gen_label = trainer.wakeTrain(sess, enc_inp, inp_len, dec_inp, outp_len, dec_outp)
            con_sen = np.concatenate((enc_inp, gen_sen), axis = 1)
            con_lab = np.concatenate((enc_label, gen_label), axis = 1)
            trainer.sleepTrain(sess, con_sen, con_len, con_label)