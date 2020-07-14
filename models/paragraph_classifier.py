import numpy as np
from gensim.models import keyedvectors

from keras import backend as K
from keras.layers import (GRU, Bidirectional, Concatenate, Dense, Dropout,
                          Embedding, Input, Lambda)
from keras.layers.merge import Concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing import sequence
from unidecode import unidecode_expect_nonascii


class ParagraphClassifier(object):
  def __init__(self, seq_maxlen=300):
    self._seq_maxlen = seq_maxlen
    self.para_classes = {
      0: 'null',
      1: 'abstract',
      2: 'intro',
      3: 'recipe',
      4: 'nonrecipe_methods',
      5: 'results',
      6: 'conclusions',
      7: 'caption'
    }

  def build_nn_model(self, nn_model=None, rnn_size=128):
    self._load_embeddings()

    input_word_ids = Input(shape=(self._seq_maxlen,))
    paragraph_position = Input(shape=(1,))

    emb_matrix =  Embedding(
      input_dim=self.emb_weights.shape[0],
      output_dim=self.emb_weights.shape[1],
      input_length=self._seq_maxlen,
      weights=[self.emb_weights],
      trainable=False,
      mask_zero=True
      )

    emb_word = emb_matrix(input_word_ids)
    drop_1 = Dropout(0.25)(emb_word)
    rnn_1 = Bidirectional(GRU(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=False))(drop_1) 

    merge_1 = Concatenate()([paragraph_position, rnn_1])
    dense_out = Dense(len(self.para_classes), activation="softmax")(merge_1)

    self.model = Model(inputs=[input_word_ids, paragraph_position], outputs=[dense_out])
    self.model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True),
                  metrics=['accuracy'])

  def featurize(self, text):
    text = self._normalize_string(text)
    tokens = self.nlp(text)
    emb_vector = []

    for token in tokens:
      tok = token.lemma_
      if tok in self.emb_vocab:
        emb_vector.append(self.emb_vocab[tok])
      else:
        emb_vector.append(1)

    return np.array(emb_vector)

  def _load_embeddings(self, fpath='bin/fasttext_embeddings-MINIFIED.model'):
    embeddings = keyedvectors.KeyedVectors.load(fpath)
    embeddings.bucket = 2000000
    self.emb_vocab = dict([('<null>', 0), ('<oov>', 1)] +
                          [(k, v.index+2) for k, v in embeddings.vocab.items()])
    self.emb_weights = np.vstack([np.zeros((1,100)), np.ones((1,100)), np.array(embeddings.syn0)])

  def train(self, X_train, Y_train, batch_size=16, num_epochs=20, verbosity=1):
    self.model.fit(
      x=X_train,
      y=Y_train,
      batch_size=batch_size,
      epochs=num_epochs,
      verbose=verbosity
    )

  def predict_one(self, paragraph_text, paragraph_position, section_text, supsection_text):
    paragraph_feature_vector = self.featurize(paragraph_text)
    padded_vec = sequence.pad_sequences([paragraph_feature_vector], maxlen=self._seq_maxlen, padding='post', truncating='post')

    return self.para_classes[np.argmax(self.fast_predict([
      padded_vec, np.array(paragraph_position).reshape(1, -1), 
      0])[0][0])]

  def predict(self, input_matrix):
    return self.fast_predict(input_matrix + [0])

  def save(self, filename):
    self.model.save(filename)

  def load(self, filename):
    self.model = load_model(filename)
    self.fast_predict = K.function(
         self.model.inputs + [K.learning_phase()],
         [self.model.layers[-1].output]
         )
    self._load_embeddings()
    
  def _normalize_string(self, string):
    ret_string = ''
    for char in string:
      if re.match('[Α-Ωα-ωÅ]', char) is not None:
        ret_string += char
      else:
        ret_string += unidecode_expect_nonascii(char)
        
    return ' '.join(ret_string.split())
