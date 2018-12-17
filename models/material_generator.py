import numpy as np

from keras import backend as K
from keras import metrics
from keras.layers import (GRU, Conv1D, Dense, Embedding, Flatten, Input,
                          Lambda, RepeatVector, TimeDistributed)
from keras.layers.merge import Concatenate, Subtract
from keras.models import Model
from keras.optimizers import Adam


class MaterialGenerator(object):
    def __init__(self):
        self.vae = None
        self.encoder = None
        self.decoder = None

    def build_nn_model(self, element_dim=103,
                       conv_window=3, conv_filters=64, 
                       rnn_dim=64, recipe_latent_dim=8,
                       intermediate_dim=64, latent_dim=8,
                       max_material_length=10, charset_size=50,):

        self.latent_dim = latent_dim
        self.recipe_latent_dim = recipe_latent_dim
        self.original_dim = max_material_length * charset_size

        x_mat = Input(shape=(max_material_length, charset_size), name="material_in")
        conv_x1 = Conv1D(conv_filters, conv_window, padding="valid", activation="relu", name='conv_enc_1')(x_mat)
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
        c_element = Input(shape=(element_dim,), name="cond_element_in")
        c_latent_recipe = Input(shape=(recipe_latent_dim,), name="cond_latent_recipe_in")

        z_conditional = Concatenate(name="concat_cond")([z, c_latent_recipe, c_element])

        decoder_h = Dense(intermediate_dim, activation="relu", name="hidden_dec")
        decoder_h_repeat = RepeatVector(max_material_length, name="h_rep_dec")
        decoder_h_gru_1 = GRU(rnn_dim, return_sequences=True, name="recurrent_dec_1")
        decoder_h_gru_2 = GRU(rnn_dim, return_sequences=True, name="recurrent_dec_2")
        decoder_h_gru_3 = GRU(rnn_dim, return_sequences=True, name="recurrent_dec_3")
        decoder_mat = TimeDistributed(Dense(charset_size, activation='softmax'), name="means_material_dec")
        
        h_decoded = decoder_h(z_conditional)
        h_decode_repeat = decoder_h_repeat(h_decoded)
        gru_h_decode_1 = decoder_h_gru_1(h_decode_repeat)
        gru_h_decode_2 = decoder_h_gru_2(gru_h_decode_1)
        gru_h_decode_3 = decoder_h_gru_3(gru_h_decode_2)
        x_decoded_mat = decoder_mat(gru_h_decode_3)

        def vae_xent_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            rec_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return rec_loss + kl_loss

        encoder = Model(inputs=[x_mat], outputs=[z_mean])

        decoder_x_input = Input(shape=(latent_dim,))

        if use_conditionals:
            decoder_inputs = Concatenate(name="concat_cond_dec")([decoder_x_input, c_latent_recipe, c_element])
        else:
            decoder_inputs = decoder_x_input
        _h_decoded = decoder_h(decoder_inputs)
        _h_decode_repeat = decoder_h_repeat(_h_decoded)
        _gru_h_decode_1 = decoder_h_gru_1(_h_decode_repeat)
        _gru_h_decode_2 = decoder_h_gru_2(_gru_h_decode_1)
        _gru_h_decode_3 = decoder_h_gru_3(_gru_h_decode_2)
        _x_decoded_mat = decoder_mat(_gru_h_decode_3)
        
        decoder = Model(inputs=[decoder_x_input, c_latent_recipe, c_element], 
                        outputs=[_x_decoded_mat])

        vae = Model(inputs=[x_mat, c_latent_recipe, c_element],
                    outputs=[x_decoded_mat])

        vae.compile(
            optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True),
            loss=vae_xent_loss,
            metrics=['categorical_accuracy']
        )

        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder

    def train(self, inputs, outputs, epochs=200, val_data=None, val_split=0.0, batch_size=16):
        fitting_results = self.vae.fit(
            x=inputs,
            y=outputs,
            epochs=epochs,
            validation_data=val_data,
            validation_split=val_split,
            batch_size=batch_size
        )

        return fitting_results.history

    def save_models(self, model_variant="default", save_path="bin"):
        self.vae.save_weights(f"{save_path}/{model_variant}_mat_gen_vae.h5")
        self.encoder.save_weights(f"{save_path}/{model_variant}_mat_gen_encoder.h5")
        self.decoder.save_weights(f"{save_path}/{model_variant}_mat_gen_decoder.h5")

    def load_models(self, model_variant="default", load_path="bin"):
        self.vae.load_weights(f"{load_path}/{model_variant}_mat_gen_vae.h5")
        self.encoder.load_weights(f"{load_path}/{model_variant}_mat_gen_encoder.h5")
        self.decoder.load_weights(f"{load_path}/{model_variant}_mat_gen_decoder.h5")
