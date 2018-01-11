import tensorflow as tf

class NeuralMT:
    def __init__(self,source_embed_size,source_vocab_size,
                      target_embed_size,target_vocab_size,
                      source_max_len, target_max_len,
                      state_units,
                      batch_size
                    ):
        self.source_embed_size = source_embed_size
        self.source_vocab_size = source_vocab_size
        self.target_embed_size = target_embed_size
        self.target_vocab_size = target_vocab_size
        self.project_to_target = tf.layers.dense(units = target_vocab_size, use_bias = False)
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.state_units = state_units
        self.batch_size = batch_size
        self.build_graph()

    def build_graph(self):
        self.source_word_embeddings = tf.Variable(tf.random_uniform(
                    [self.source_vocab_size,self.source_embed_size],-1,1), tf.int64)
        self.target_word_embeddings = tf.Variable(tf.random_uniform(
                    [self.target_vocab_size,self.target_embed_size],-1,1), tf.int64)
        # for time major, encoder_inps size [source_max_len,batch_size]
        # and encoder_emb_inp size [source_max_len,batch_size,source_embed_size]
        # so as decoder_inps and decoder_emb_inp
        self.encoder_inps = tf.placeholders([self.source_max_len,self.batch_size],tf.int64,"encoder_inps")

        # need to be figure out
        self.decoder_inps = tf.placeholders([self.target_max_len,self.batch_size],tf.int64,"decoder_inps")
        self.decoder_outps = tf.placeholders([self.target_max_len,self.batch_size],tf.int64,"decoder_outps")

        source_seq_len = tf.placeholders([None],tf.int64)
        target_seq_len = tf.placeholders([None],tf.int64)
        encoder_emb_inp = tf.embeddings_lookup(self.source_word_embeddings,self.encoder_inps)
        decoder_emb_inp = tf.embeddings_lookup(self.target_word_embeddings,self.decoder_inps)
        encoder_outputs, encoder_state = build_encoder(encoder_emb_inp,source_seq_len)
        logits = self.build_decoder(decoder_emb_inp,encoder_outputs,encoder_state, target_seq_len)
        target_weights = tf.sequence_mask(target_seq_len, target_max_len, dtype=logits.dtype)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.decoder_outps, logits = logits)
        self.train_loss = (tf.reduce_sum(cross_entropy * target_weight) / self.batch_size)

    def build_encoder(self, encoder_emb_inp, source_seq_len):
        # encoder_outputs: [source_max_len, batch_size, state_units]
        # encoder_state: [batch_size, state_units]
        encode_cell = tf.nn.rnn_cell.BasicLSTMCell(self.encoder.units)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp,
                                            sequence_length = source_seq_len,
                                            time_major = True)
        return encoder_outputs, encoder_state

    def build_decoder(self, decoder_emb_inp, encoder_outputs, encoder_state, source_seq_len, target_seq_len):
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_units)
        # attention_states: [batch_size,source_max_len,state_units]
        # note encoder and decoder have to have same size state_units
        # to calculate score in performing attention
        attention_states = tf.transpose(encoder_outputs,[1,0,2])
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.state_units,
                                                                attention_states,
                                                                memory_sequence_length = source_seq_len)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                           attention_layer_size = self.state_units)
        helper = tf.contrib.seq2seq.TrainingHelper(inputs = decoder_emb_inp,
                                                   sequence_length = target_seq_len,
                                                   time_major = True)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                                  initial_state = encoder_state,
                                                  output_layer = self.project_to_target)
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        logits = outputs.rnn_output
        hightest_indice = outputs.sample_id

        return logits
