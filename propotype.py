
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense



#decoder_emb_inp = tf.get_variable("decoder_inp",[1,5,10],initializer = initializer)
tgt_sos_id = 0
tgt_eos_id = 1

l = open("word.txt").read().split()
vocab_size = len(l)
print(vocab_size)
print(l)

test_len = 6
source_len = 7
embed_size = 5
batch_sent = tf.placeholder(tf.int32, shape = [None, None])
input_sent = tf.placeholder(tf.int32, shape = [None, None])

hid_units = 5

initializer = tf.random_uniform_initializer(-0.1, 0.1)

word_embeddings = tf.Variable(tf.random_uniform([vocab_size,embed_size],1,-1))

sent_embed = tf.nn.embedding_lookup(word_embeddings, batch_sent)
sour_embed = tf.nn.embedding_lookup(word_embeddings, input_sent)

# encoder
shit_state = tf.contrib.rnn.LSTMStateTuple(tf.Variable(tf.random_normal([1,hid_units], mean=-1, stddev=4)),
                 							  tf.Variable(tf.random_normal([1,hid_units], mean=-1, stddev=4)))

encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hid_units)
encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell, sour_embed, initial_state = shit_state)


print(sent_embed.shape)



"""
attention_states = tf.Variable(tf.random_uniform([1,10,hid_units],1,-1))
"""
attention_states = encoder_output
attention_mechanism = tf.contrib.seq2seq.LuongAttention(hid_units,
                                                        attention_states)
                                                        #memory_sequence_length = 1) # len of attention_states = source_seq_len

decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hid_units)

decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
												   initial_cell_state = encoder_state,
                                                   attention_layer_size = hid_units)

helper = tf.contrib.seq2seq.TrainingHelper(inputs = sent_embed,
                                           sequence_length = [6]) # sequence_length is the decoder sequence lengh
greedyHelper = tf.contrib.seq2seq.GreedyEmbeddingHelper(word_embeddings,tf.fill([1],tgt_sos_id),tgt_eos_id)
															  # could be less than decoder input lengh but not more than decoder input lengh
initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size = 1)
# greedy helper is set up
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, greedyHelper,
                                          initial_state = initial_state,
                                          output_layer = Dense(vocab_size))

outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,maximum_iterations = 10)
#

logits = outputs.rnn_output
print(logits.shape)
hightest_indice = outputs.sample_id


with tf.Session() as sess:


    word2id = {w:i for i,w in enumerate(l)}
    id2word = {i:w for i,w in enumerate(l)}

    sent = "the web Google does presume logarithmic month If presume maximum PR the we scale"
    source_sent = "we presume the scale is pages all"

    source_word = source_sent.split()
    source_id = [[word2id[w] for w in source_word]]


    sent_word = sent.split()
    sent_id = [word2id[w] for w in sent_word]
    test_sent = np.array([sent_id])

    sess.run(tf.global_variables_initializer())
    outputs, ind = sess.run([logits,hightest_indice], feed_dict = {batch_sent : test_sent, input_sent : source_id})

    print(outputs.shape)
    print("ind", ind)

    for sent in ind:
        print(' '.join([id2word[id] for id in sent]))
