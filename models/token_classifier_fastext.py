import json
import spacy
from spacy.tokens import Doc
import re
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding, GRU, TimeDistributed, Bidirectional, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
from gensim.models import keyedvectors
from unidecode import unidecode_expect_nonascii
import numpy as np
import pandas as pd

class TokenClassifier(object):
    def __init__(self, nlp=None, seq_maxlen=100, fpath='data/ner_annotations_split.json'):
        if nlp is None:
            self.nlp = spacy.load('en_core_web_sm')
        else:
            self.nlp = nlp
        self._seq_maxlen = seq_maxlen
        self.token_classes = {
            0: "null",
            1: "nonrecipe-material",
            2: "unspecified-material",
            3: "material",
            4: "precursor",
            5: "solvent",
            6: "gas", 
            7: "target",
            8: "number",
            9: "amount-unit",
            10: "amount-misc",
            11: "condition-unit",
            12: "condition-misc",
            13: "condition-type",
            14: "property-unit",
            15: "property-misc",
            16: "property-type",
            17: "synthesis-apparatus",
            18: "apparatus-property-type",
            19: "apparatus-descriptor",
            20: "apparatus-unit",
            21: "brand", 
            22: "reference",
            23: "operation",
            24: "meta",
            25: "material-descriptor"
        }
        self.inv_token_classes = {v: k for k, v in self.token_classes.items()}
    
        self._load_embeddings()
        annotated_data = json.loads(open(fpath, "r").read())
        self.X_train, self.X_dev, self.X_test = [],[],[]
        self.y_train, self.y_dev, self.y_test = [],[],[]
        for ann_paper in annotated_data["data"]:
            for i, (sent, labels) in enumerate(zip(ann_paper["tokens"], ann_paper["labels"])):
                if i == 0: continue #skip titles
                ft_vec = self.featurize(sent)
                onehot_labels = np.zeros(shape=(self._seq_maxlen, len(self.token_classes)))
                for j, label in enumerate(labels[:self._seq_maxlen]):
                    onehot_label = [0.0]*len(self.token_classes)
                    onehot_label[self.inv_token_classes[label]] = 1.0
                    onehot_labels[j] = onehot_label
                if ann_paper["split"] == "train":
                    self.X_train.append(ft_vec)
                    self.y_train.append(onehot_labels)
                elif ann_paper["split"] == "dev":
                    self.X_dev.append(ft_vec)
                    self.y_dev.append(onehot_labels)
                else:
                    self.X_test.append(ft_vec)
                    self.y_test.append(onehot_labels)
        self.X_train = np.asarray(sequence.pad_sequences(self.X_train, maxlen=self._seq_maxlen, padding='post', truncating='post'))
        self.X_dev = np.asarray(sequence.pad_sequences(self.X_dev, maxlen=self._seq_maxlen, padding='post', truncating='post'))
        self.X_test = np.asarray(sequence.pad_sequences(self.X_test, maxlen=self._seq_maxlen, padding='post', truncating='post'))
        self.y_train, self.y_dev, self.y_test = np.asarray(self.y_train), np.asarray(self.y_dev), np.asarray(self.y_test)

        print('Initialized Fasttext Token Model.....')
        print('Train Set Shape: Input-', self.X_train.shape, ' Output-', self.y_train.shape)
        print('Dev Set Shape: Input-', self.X_dev.shape, ' Output-', self.y_dev.shape)
        print('Test Set Shape: Input-', self.X_test.shape, ' Output-', self.y_test.shape)

    def build_nn_model(self, recurrent_dim=256, focal=False):
        try:
            x = self.emb_vocab_w2v
        except:
            self._load_embeddings()

        emb_ft = Embedding(
            input_dim=self.emb_weights_ft.shape[0],
            output_dim=self.emb_weights_ft.shape[1],
            input_length=self._seq_maxlen,
            weights=[self.emb_weights_ft],
            trainable=False,
            mask_zero=True
        )(input_ft_ids)

        drop_1 = Dropout(0.1)(emb_ft)
        rnn_1 = Bidirectional(GRU(recurrent_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(drop_1)
        dense_out = TimeDistributed(Dense(len(self.token_classes), activation="softmax"))(rnn_1)

        model = Model(inputs=[input_ft_ids], outputs=[dense_out])

        model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                    metrics=['accuracy'])

        self.model = model
        self.fast_predict = K.function(
            self.model.inputs + [K.learning_phase()],
            [self.model.layers[-1].output]
        )
    
    def featurize(self, words):
        ft_vector = []
        
        words = [self._normalize_string(w) for w in words]
        
        spacy_doc = Doc(self.nlp.vocab, words=words)
        self.nlp.tagger(spacy_doc)
        self.nlp.parser(spacy_doc)
        self.nlp.entity(spacy_doc)
        spacy_tokens = spacy_doc[:self._seq_maxlen]

        for word_tok in spacy_tokens:
            word_string = word_tok.lemma_

            if word_string in self.emb_vocab_ft:
                ft_vector.append(self.emb_vocab_ft[word_string])
            else:
                ft_vector.append(1)

        return np.array(ft_vector)
    
    def train(self, batch_size=256, num_epochs=30, checkpt_filepath=None, 
            checkpt_period=5, stop_early=False, verbosity=1, val_split=0.0):

        callbacks = []
        callbacks.append(schedule)
        if checkpt_filepath is not None:
            callbacks = [
                ModelCheckpoint(
                checkpt_filepath,
                monitor='val_loss',
                verbose=0,
                save_best_only=True,
                period=checkpt_period
                ),
                schedule
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
            verbose=verbosity
        )

    def test(self, confusion_matrix=True):
        raw_preds_test = self.model.predict(self.X_test)
        train_labels, train_predictions, train_words = [],[],[]
        test_labels, test_predictions, test_words = [],[],[]
        num_totally_correct = 0
        for i, (sent, labels, preds) in enumerate(zip(self.test_sentences, self.Y_test, raw_preds_test)):
            all_correct = True
            for word, label, pred in zip(sent, labels, preds):
                test_words.append(word)
                test_labels.append(self.token_classes[np.argmax(label)])
                test_predictions.append(self.token_classes[np.argmax(pred)])
                if self.token_classes[np.argmax(pred)] != self.token_classes[np.argmax(label)]:
                    all_correct = False
            if all_correct:
                num_totally_correct+=1
        print('Test Set Results.....')
        print(classification_report(test_labels, test_predictions))
        if confusion_matrix:
            print('---')
            print(confusion_matrix(test_labels, test_predictions))
        print('---')
        print('Completely Correct Sentences =', num_totally_correct, len(self.test_sentences), round(num_totally_correct/len(self.test_sentences),2))
            
        return self.model.predict(self.X_test)

    def evaluate(self, batch_size=32):
        return self.model.evaluate(self.X_test, self.Y_test, batch_size=batch_size)
    
    def predict_many(self, tokenized_sentences):
        predictions = []
        for sent in tokenized_sentences:
            predictions.append(self.predict_one(sent))
        return predictions
    
    def predict_one(self, words):
        num_words = len(words)
        ft_feature_vector = self.featurize(words)
        ft_feature_vector = sequence.pad_sequences([ft_feature_vector], maxlen=self._seq_maxlen, 
                                                    padding='post', truncating='post')

        return [self.token_classes[np.argmax(w)] for w in self.fast_predict([ft_feature_vector, 0])[0][0]][:num_words]
    
    def save(self, filepath='bin/token_classifier.model'):
        self.model.save(filepath)

    def load(self, filepath='bin/token_classifier.model'):
        self.model = load_model(filepath)
        self._load_embeddings()
        self.fast_predict = K.function(
            self.model.inputs + [K.learning_phase()],
            [self.model.layers[-1].output]
        )
    
    def _load_embeddings(self, ft_fpath='bin/fasttext_embeddings-MINIFIED.model'):
        ft_embeddings = keyedvectors.KeyedVectors.load(ft_fpath)
        self.emb_vocab_ft = dict([('<null>', 0), ('<oov>', 1)] +
                         [(k, v.index+2) for k, v in ft_embeddings.vocab.items()])
        self.emb_weights_ft = np.vstack([np.zeros((1,100)), np.ones((1,100)), np.array(ft_embeddings.syn0)])

    def _normalize_string(self, string):
        ret_string = u''
        for char in string:
            if re.match(u'[Α-Ωα-ωÅ]', char) is not None:
                ret_string += str(char)
            else:
                ret_string += str(unidecode_expect_nonascii(char))
            
        return ret_string