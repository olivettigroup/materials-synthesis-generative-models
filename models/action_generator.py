import numpy as np

from keras import backend as K
from keras import metrics
from keras.layers import (GRU, Conv1D, Dense, Flatten, Input,
                          Lambda, RepeatVector, TimeDistributed)
from keras.layers.merge import Concatenate, Subtract
from keras.models import Model
from keras.optimizers import Adam


class ActionGenerator(object):
    def __init__(self):
        self.vae = None
        self.encoder = None
        self.decoder = None
        self.op_emb_vocab = {}

    def build_nn_model(self, max_recipe_steps=20, 
                       rnn_dim=64, conv_window=5, conv_filters=128,
                       intermediate_dim=128, latent_dim=8, material_dim=100):

        self.latent_dim = latent_dim
        self.original_dim = max_recipe_steps * len(self.op_emb_vocab)

        if not self.op_emb_vocab:
            # Fill vocab with dummy variables if you want to build 
            # an empty model to visualize/draw its architecture
            self.op_emb_vocab = {i:i for i in range(50)}

        x_ops = Input(shape=(max_recipe_steps, len(self.op_emb_vocab)), name="ops_in")

        conv_x1 = Conv1D(conv_filters, conv_window, padding="valid", activation="relu", name='conv_enc_1')(x_ops)
        conv_x2 = Conv1D(conv_filters, conv_window, padding="valid", activation="relu", name='conv_enc_2')(conv_x1)
        conv_x3 = Conv1D(conv_filters, conv_window, padding="valid", activation="relu", name='conv_enc_3')(conv_x2)
        h_flatten = Flatten()(conv_x3)
        h = Dense(intermediate_dim, activation="relu", name="hidden_enc")(h_flatten)

        z_mean_func = Dense(latent_dim, name="means_enc")
        z_log_var_func = Dense(latent_dim, name="vars_enc")

        z_mean = z_mean_func(h)
        z_log_var = z_log_var_func(h)

        def sample(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(latent_dim,), mean=0.0, stddev=1.0)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z = Lambda(sample, name="lambda_sample")([z_mean, z_log_var])
        c_material = Input(shape=(material_dim,), name="cond_matrl_in")

        z_conditional = Concatenate(name="concat_cond")([z, c_material])

        decoder_h = Dense(intermediate_dim, activation="relu", name="hidden_dec")
        decoder_h_repeat = RepeatVector(max_recipe_steps, name="h_rep_dec")
        decoder_h_gru_1 = GRU(rnn_dim, return_sequences=True, name="recurrent_dec_1")
        decoder_h_gru_2 = GRU(rnn_dim, return_sequences=True, name="recurrent_dec_2")
        decoder_h_gru_3 = GRU(rnn_dim, return_sequences=True, name="recurrent_dec_3")
        decoder_ops = TimeDistributed(Dense(len(self.op_emb_vocab), activation='softmax'), name="means_op_dec")
        
        h_decoded = decoder_h(z_conditional)
        h_decode_repeat = decoder_h_repeat(h_decoded)
        gru_h_decode_1 = decoder_h_gru_1(h_decode_repeat)
        gru_h_decode_2 = decoder_h_gru_2(gru_h_decode_1)
        gru_h_decode_3 = decoder_h_gru_3(gru_h_decode_2)
        x_decoded_ops = decoder_ops(gru_h_decode_3)

        def vae_xent_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            rec_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return rec_loss + kl_loss

        encoder = Model(inputs=[x_ops], outputs=[z_mean])

        decoder_x_input = Input(shape=(latent_dim,))
        decoder_inputs = Concatenate(name="concat_cond_dec")([decoder_x_input, c_material])

        _h_decoded = decoder_h(decoder_inputs)
        _h_decode_repeat = decoder_h_repeat(_h_decoded)
        _gru_h_decode_1 = decoder_h_gru_1(_h_decode_repeat)
        _gru_h_decode_2 = decoder_h_gru_2(_gru_h_decode_1)
        _gru_h_decode_3 = decoder_h_gru_3(_gru_h_decode_2)
        _x_decoded_ops = decoder_ops(_gru_h_decode_3)
        
        decoder = Model(inputs=[decoder_x_input, c_material], 
                        outputs=[_x_decoded_ops])

        vae = Model(inputs=[x_ops, c_material],
                    outputs=[x_decoded_ops])

        vae.compile(
        optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True),
        loss=[vae_xent_loss],
        metrics=['categorical_accuracy']
        )

        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder

    def train(self, inputs, outputs, epochs=300, val_data=None, val_split=0.0, batch_size=16):
        fitting_results = self.vae.fit(
            x=inputs,
            y=outputs,
            epochs=epochs,
            validation_split=val_split,
            validation_data=val_data,
            batch_size=batch_size
        )

        return fitting_results.history

    def set_embeddings(self, operation_vocab):   
        self.op_emb_vocab = operation_vocab

    def save_models(self, model_variant="default", save_path="bin/"):
        self.vae.save_weights(save_path + model_variant + "_seq_gen_vae.h5")
        self.encoder.save_weights(save_path + model_variant + "_seq_gen_encoder.h5")
        self.decoder.save_weights(save_path + model_variant + "_seq_gen_decoder.h5")

    def load_models(self, model_variant="default", load_path="bin/"):
        self.vae.load_weights(load_path + model_variant + "_seq_gen_vae.h5")
        self.encoder.load_weights(load_path + model_variant + "_seq_gen_encoder.h5")
        self.decoder.load_weights(load_path  + model_variant + "_seq_gen_decoder.h5")
