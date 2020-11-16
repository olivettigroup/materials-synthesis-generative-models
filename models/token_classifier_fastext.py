import json


class TokenClassifier(object):
    def __init__(self, nlp=None, seq_maxlen=100, data='data/ner_annotations_split.json'):
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
            if ann_paper["split"] = "train":
                self.X_train.append(ft_vec)
                self.y_train.append(onehot_labels)
            elif ann_paper["split"] = "dev":
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
