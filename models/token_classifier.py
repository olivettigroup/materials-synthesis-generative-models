import json

import numpy as np

import tensorflow as tf
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from keras import backend as K
from keras import optimizers
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.layers import (GRU, Bidirectional, Dense, Dropout, Input,
                          TimeDistributed)
from keras.models import Model, load_model


class TokenClassifier(object):
  def __init__(self, seq_maxlen=100, vocab="bin/vocab.txt", 
                      options="bin/elmo_options.json", 
                      weights="bin/elmo_weights.hdf5", use_cpu=False, fpath='data/ner_annotations_split.json'):
    self.token_classes = {
      0: "null",
      1: "precursor",
      2: "target",
      3: "operation",
    }
    self.session = None
    self.inv_token_classes = {v: k for k, v in self.token_classes.items()}
    self._seq_maxlen = seq_maxlen
    self.use_cpu = use_cpu
    self._load_tf_session(use_cpu=use_cpu)
    self._load_embeddings(vocab, options, weights)
    
    train_sentences, dev_sentences, test_sentences = [],[],[]
    train_labels, dev_labels, test_labels = [],[],[]
    for paper in data['data']:
      if paper['split'] == 'train':
        train_sentences.extend(paper['tokens'][1:]) # first "sentence" is the title which we don't want right now
        train_labels.extend(paper['labels'][1:])
      elif paper['split'] == 'dev':
        dev_sentences.extend(paper['tokens'][1:])
        dev_labels.extend(paper['labels'][1:])
      else:
        test_sentences.extend(paper['tokens'][1:])
        test_labels.extend(paper['labels'][1:])
    print('Initializing ELMO Token Model.....')
    train_elmo_features = token_classifier.featurize_elmo_list(train_sentences)
    print('Train Input shape:', train_elmo_features.shape)
    dev_elmo_features = token_classifier.featurize_elmo_list(dev_sentences)
    print('Dev Input shape:', dev_elmo_features.shape)
    test_elmo_features = token_classifier.featurize_elmo_list(test_sentences)
    print('Test Input shape:', test_elmo_features.shape)
    y_train, y_dev, y_test = [],[],[]
    for labels in train_labels:
      train_onehot_labels = np.zeros(shape=(token_classifier._seq_maxlen, len(token_classifier.token_classes)))
      for j, label in enumerate(labels[:token_classifier._seq_maxlen]):
        if label not in ['precursor', 'target', 'operation']:
          label = 'null'
        train_onehot_label = [0.0]*len(token_classifier.token_classes)
        train_onehot_label[token_classifier.inv_token_classes[label]] = 1.0
        train_onehot_labels[j] = train_onehot_label
      y_train.append(train_onehot_labels)
    for labels in dev_labels:
      dev_onehot_labels = np.zeros(shape=(token_classifier._seq_maxlen, len(token_classifier.token_classes)))
      for j, label in enumerate(labels[:token_classifier._seq_maxlen]):
        if label not in ['precursor', 'target', 'operation']:
            label = 'null'
        dev_onehot_label = [0.0]*len(token_classifier.token_classes)
        dev_onehot_label[token_classifier.inv_token_classes[label]] = 1.0
        dev_onehot_labels[j] = dev_onehot_label
      y_dev.append(dev_onehot_labels)
    for labels in test_labels:
      test_onehot_labels = np.zeros(shape=(token_classifier._seq_maxlen, len(token_classifier.token_classes))) 
      for j, label in enumerate(labels[:token_classifier._seq_maxlen]):
        if label not in ['precursor', 'target', 'operation']:
            label = 'null'
        test_onehot_label = [0.0]*len(token_classifier.token_classes)
        test_onehot_label[token_classifier.inv_token_classes[label]] = 1.0
        test_onehot_labels[j] = test_onehot_label
      y_test.append(test_onehot_labels)
    y_test = np.array(y_test)
    y_dev = np.array(y_dev)
    y_train = np.array(y_train)
    print('Train Output Shape:', y_train.shape)
    print('Dev Output Shape:', y_dev.shape)
    print('Test Output Shape:', y_test.shape)

    self.X_train = train_elmo_features
    self.X_dev = dev_elmo_features
    self.X_test = test_elmo_features
    self.Y_train = y_train
    self.Y_dev = y_dev
    self.Y_test = y_test

  def build_nn_model(self, recurrent_dim=2048, dense1_dim=1024, elmo_dim=1024):
    input_vectors = Input(shape=(self._seq_maxlen, elmo_dim))
    drop_1 = Dropout(0.5)(input_vectors)
    rnn_1 = Bidirectional(GRU(recurrent_dim, return_sequences=True))(drop_1)
    drop_2 = Dropout(0.5)(rnn_1)
    dense_1 = TimeDistributed(Dense(dense1_dim, activation="relu"))(drop_2)
    drop_3 = Dropout(0.5)(dense_1)
    dense_out = TimeDistributed(Dense(len(self.token_classes), activation="softmax"))(drop_3)

    model = Model(inputs=[input_vectors], outputs=[dense_out])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True),
                  metrics=['accuracy'])
    self.model = model
    self.fast_predict = K.function(
      self.model.inputs + [K.learning_phase()],
      [self.model.layers[-1].output]
    )

  def train(self, batch_size=256, num_epochs=30, checkpt_filepath=None, 
            checkpt_period=5, verbosity=1, val_split=0.0, stop_early=False):
    self._load_tf_session(use_cpu=self.use_cpu)
    callbacks = [
      ModelCheckpoint(
        checkpt_filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        period=checkpt_period
        )
    ]
    if stop_early:
      callbacks.append(
        EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')
      )

    self.model.fit(
      x=self.X_train,
      y=self.Y_train,
      batch_size=batch_size,
      epochs=num_epochs,
      validation_split=val_split,
      validation_data=(self.X_dev, self.Y_dev),
      callbacks= callbacks,
      verbose=verbosity,
    )


  def featurize_elmo_list(self, sent_toks_list, batch_size=128):
    padded_list = [t[:self._seq_maxlen] + ['']*max(self._seq_maxlen - len(t), 0) for t in sent_toks_list]

    features = []
    prev_i = 0
    for i in range(batch_size, len(sent_toks_list) + batch_size - 1, batch_size):
      context_ids = self.batcher.batch_sentences(padded_list[prev_i:i])
      elmo_features = self.session.run(
          [self.elmo_context_output['weighted_op']],
          feed_dict={self.character_ids: context_ids}
      )
      features.extend(elmo_features[0])
      prev_i = i

    return np.array(features)

  def featurize_elmo(self, sent_toks):
    length = [self._seq_maxlen]
    padded_toks = [sent_toks[:self._seq_maxlen]  + ['']*max(self._seq_maxlen - len(sent_toks), 0)]

    context_ids = self.batcher.batch_sentences(padded_toks)
    elmo_feature = self.session.run(
        [self.elmo_context_output['weighted_op']],
        feed_dict={self.character_ids: context_ids}
    )

    features = elmo_feature[0]
    return np.array(features)

  def evaluate(self, batch_size=32):
    return self.model.evaluate(self.X_test, self.Y_test, batch_size=batch_size)

  def predict_one(self, words):
    num_words = len(words)
    elmo_feature_vector = self.featurize_elmo(words)
    return [self.token_classes[np.argmax(w)] for w in self.fast_predict([elmo_feature_vector])[0].squeeze()][:num_words]

  def predict_many(self, sent_list):
    num_words = [len(s) for s in sent_list]
    elmo_feature_vectors = self.featurize_elmo_list(sent_list)
    predicted_labels = []

    for elmo_vec, sent_len in zip(elmo_feature_vectors, num_words):
      predicted_labels.append(
        [self.token_classes[np.argmax(w)] for w in self.fast_predict([[elmo_vec]])[0].squeeze()][:sent_len]
      )

    return predicted_labels

  def save(self, filepath='bin/token_classifier.model'):
    self.model.save(filepath)

  def load(self, filepath='bin/token_classifier-SNAPSHOT.model'):
    self.model = load_model(filepath)
    self.fast_predict = K.function(
      self.model.inputs + [K.learning_phase()],
      [self.model.layers[-1].output]
    )

  def _load_tf_session(self, use_cpu=False):
    if self.session is not None:
      self.session.close()

    if use_cpu:
      try:
        config = tf.ConfigProto(
          device_count={'CPU' : 1, 'GPU' : 0},
          allow_soft_placement=True,
        )
      except:
        config = tf.compat.v1.ConfigProto(
          device_count={'CPU' : 1, 'GPU' : 0},
          allow_soft_placement=True,
        )
    else:
      try:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
      except:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
    
    try:
      self.session = tf.InteractiveSession(config=config)
    except:
      self.session = tf.compat.v1.InteractiveSession(config=config)

  def _load_embeddings(self, 
                      vocab="vocab.txt", 
                      options="elmo_options.json", 
                      weights="elmo_weights.hdf5"):
    self.elmo_model = BidirectionalLanguageModel(options, weights)
    self.batcher = Batcher(vocab, 50)

    self.character_ids = tf.placeholder('int32', shape=(None, None, 50))
    context_embeddings_op = self.elmo_model(self.character_ids)
    self.elmo_context_output = weight_layers(
      'output', context_embeddings_op, l2_coef=0.0
    )

    tf.global_variables_initializer().run()
