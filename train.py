import tensorflow as tf
import os
import trainer
import pickle
from trainer import Trainer
import random
import numpy as np
from configs import args
from data import DataFlow
from testData import Testor
import sys
import matplotlib.pyplot as plt

"""
hid_units = 300
batch_size = 32
epochs = 7
c_size = 2
beam_width = 5
embed_size = 300
sos_id = 0
eos_id = 1
vocab_size = 20000
max_len = 15
model_path = './saved_X/X.ckpt'
model_dir = 'saved_X'
"""

testor = Testor()
test = testor.test_inputs
given_c = testor.test_labels


with tf.Session() as sess:

    trainer = Trainer()

    data = DataFlow()
    iterator, total_len = data.load()
    next_element = iterator.get_next()
    total_step = args.epochs * total_len / args.batch_size

    test_len = []
    for sen in test:
        test_len.append(len(sen.split()))
    test_inp = np.zeros((len(test),max(test_len)))
    for i,sen in enumerate(test):
        for j,word in enumerate(sen.split()):
            if word in data.word2id.keys():
                test_inp[i,j] = data.word2id[word]
            else:
                test_inp[i,j] = data.word2id['<unk>']

    if os.path.isdir(args.model_dir):
        print("loading model from {}".format(args.model_dir))
        trainer.saver.restore(sess,args.model_path)
        print("loaded")
        #trainer.encoder.load(sess)
        #trainer.generator.load(sess)
        #trainer.discriminator.load(sess)
    else:
        print("global initializing")
        sess.run(tf.global_variables_initializer())
        print("global initialized")

    kl_loss_list = []
    rec_loss_list = []
    kl_weight_list = []
    step_list = []

    tr_acc = []
    te_acc = []

    step = 0
    temp_step = 0
    for epoch in range(args.epochs):
        print("epoch---------{}".format(epoch))
        sess.run(iterator.initializer)
        #writer = tf.summary.FileWriter("output", sess.graph)
        #writer.close()
        while True:
            try:
                enc_inp, dec_inp, dec_tar, enc_label = sess.run(next_element)

                enc_len, dec_len = sess.run([tf.count_nonzero(enc_inp, axis = 1),
                							 tf.count_nonzero(dec_inp, axis = 1)])

                enc_label = sess.run(tf.one_hot(enc_label, depth = 2))

                if step > 4e5:

                    # VAE train

                    vae_loss, vae_rec, vae_kl, vae_sen, vae_u, vae_s, sample_c, kl_w = trainer.vaeTrain(sess,
                                     enc_inp, enc_len, dec_inp, dec_len, dec_tar,step)

                    # pre-trian discriminator with supervised label
                    #pre_loss, pre_discri_acc, supv_c = trainer.preTrain(sess, enc_inp, enc_len, enc_label)

                    if step % 20 == 0:

                        #we = sess.run(trainer.word_embed)
                        #print("step: {} we: {}".format(step,we))

                        #print("vae_gvs : {}".format(vae_gvs))
                        #inf_ids = trainer.inference(sess,test_inp, test_len, given_c)

                        #print("step: {}, vae_u: {}, vae_s: {}".format(step,kl_weight,vae_u,vae_s))
                        print("step: {}, kl_w: {:.5f}, vae_loss: {:.2f}, vae_kl: {:.2f}, vae_rec: {:.2f}".format(step,kl_w,vae_loss, vae_kl, vae_rec))
                        step_list.append(step)
                        kl_weight_list.append(kl_w)
                        kl_loss_list.append(vae_kl)
                        rec_loss_list.append(vae_rec)

                        for tr, truth in zip(inf_ids, test_inp):
                            print("step: {} ".format(step) + "tru: " + ' '.join([data.id2word[id] for id in truth]))
                            print("step: {} ".format(step) + "inf: " + ' '.join([data.id2word[id] for id in tr]))

                        """
                        #print("step: {}, pre-train loss : {}, accuracy : {}".format(step,pre_loss, pre_discri_acc))
                        for tr, truth, sv_t, spl_c in zip(vae_sen, dec_inp, enc_label, sample_c):
                            print("step: {} ".format(step) + "tru: " + ' '.join([data.id2word[id] for id in truth]) + '|| t: {}'.format(np.argmax(sv_t)))
                            print("step: {} ".format(step) + "vae: " + ' '.join([data.id2word[id] for id in tr]) + '|| sample_c: {}'.format(np.argmax(spl_c)))
                        """
                else:

                    # wake phase
                    kl_step = 6650
                    temp_step += 1
                    gen_sen, gen_label, c_loss, z_loss, kl_loss, rec_loss, syn_acc, mean, sig, \
                     pred_c, logit_encode, lgs, kl_w, temp =  trainer.wakeTrain(sess, enc_inp,
                                        enc_len, dec_inp, dec_len, dec_tar,kl_step,temp_step)

                    # sleep phase
                    sleep_loss, sleep_acc, supv_c = trainer.sleepTrain(sess,
                     enc_inp, enc_len, enc_label, gen_sen, dec_len, gen_label)

                    if step % 50 == 0:

                        inf_ids = trainer.inference(sess,test_inp, test_len, given_c)
                        for tr, truth in zip(inf_ids, test_inp):
                            print("step: {} ".format(step) + "tru: " + ' '.join([data.id2word[id] for id in truth]))
                            print("step: {} ".format(step) + "inf: " + ' '.join([data.id2word[id] for id in tr]))

                        #tr_acc.append(sleep_acc)
                        #we = sess.run(trainer.word_embed)
                        #print("step: {} we: {}".format(step,we))
                        #print("step: {}, lgs: {}".format(step, lgs))
                        #print("step: {}, logit encode: {}".format(step, logit_encode))
                        #print("step: {}, mean: {}, sig: {}".format(step, mean, sig))
                        print("step: {}, kl_weight: {}, c_loss: {}, z_loss: {}".format(step,kl_w,c_loss,z_loss))
                        print("step: {}, kl_loss: {} rec_loss: {}, syn_acc: {}".format(step,kl_loss,rec_loss, syn_acc))
                        print("step: {}, sleep loss : {}, accuracy : {}, temp: {}".format(step,sleep_loss,sleep_acc,temp))
                        """
                        for tr, truth, sv_t, sv_c, gen_c, p_c in zip(gen_sen, dec_tar,enc_label, supv_c, gen_label, pred_c):
                            print("step: {} ".format(step) + "tru: " + ' '.join([data.id2word[id] for id in truth]) + ' sl_c: {} t: {}'.format(sv_c,np.argmax(sv_t)))
                            print("step: {} ".format(step) + "tra: " + ' '.join([data.id2word[id] for id in tr]) + ' wk_c: {} s: {}'.format(p_c, gen_c))
                        """

                step += 1
            except tf.errors.OutOfRangeError:
                s_path = trainer.saver.save(sess, args.wake_path)
                print("Model saved in file: %s" % args.wake_path)
                #plt.plot(step_list,kl_weight_list,label = 'kl_weight')
                """
                plt.plot(step_list,rec_loss_list,label = 'reconst loss')
                plt.plot(step_list,kl_loss_list,label = 'kl_loss')
                pickle.dump(step_list,open('step_list.pkl','wb'))
                pickle.dump(rec_loss_list,open('rec_loss_list.pkl','wb'))
                pickle.dump(kl_loss_list,open('kl_loss_list.pkl','wb'))
                pickle.dump(kl_weight_list,open('kl_weight_list.pkl','wb'))
                plt.legend()
                plt.show()
                """
                break
