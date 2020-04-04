# Generative models and NLP resources for materials synthesis

Public release of data and code for materials synthesis generation, along with NLP resources for materials science. 🎉

This code and data is a companion to the paper, "**Inorganic Materials Synthesis Planning with Literature-Trained Neural Networks**."

## Demo 🐍

`demo.ipynb` (or `demo.html`) contains a Python demo showcasing the fine-tuned word embeddings introduced in this paper. The demo also provides an example of building and inspecting the autoencoder models.

## Annotated NER Data 📝

`data/ner_annotations.json` contains tokenized and labelled NER information for 235 synthesis recipes. Each annotated recipe is marked by a `"split"` key which may be `"train"`, `"test"`, or `"dev"` - and there are also five papers (which were used for interannotator agreement internally) marked with a `"metrics"` split. These splits are merely suggested (and were indeed computed randomly), and so we encourage others to use whatever splits of the data they deem appropriate. This file should be usable as-is for training NER models. Each annotated document contains equal-length arrays of `tokens` and their respective `labels`.

`data/brat/` contains raw annotation files in the BRAT annotation format. You can load these into your own instance of BRAT and modify the annotations however you like! These files contain event/relation annotations as well (e.g., "heat" acts on "titania").

## NLP Resource Downloads 💽

Along with this work, we also open-source two pre-trained word embedding models: FastText and ELMo, each trained on our internal database of over 2.5 million materials science articles.

The FastText model follows the `gensim` Python library, and can be loaded as a `keyedvectors` object. Please see the `gensim` documentation for more details. Note that our version FastText is trained on **lowercase** text only.

The ELMo model follows the weights/options layout in the `allenai/bilm-tf` public GitHub repository. You can load the embeddings as described in their `README` (or just use the code in this repo, at `models/token_classifier.py`), but simply swap out the weight and options files. We found that using the default `vocab.txt` works fine, so there's no need to swap anything out in that case. As per the recommendations of the ELMo authors, we **don't perform lowercase normalization** for ELMo, so you can compute word vectors for text "as-is."

Links to the trained models/weights are as follows:

- FastText: https://figshare.com/s/70455cfcd0084a504745
- ELMo: https://figshare.com/s/ec677e7db3cf2b7db4bf

## Neural Network Models/Data 🧠

`models/action_generator.py` contains the architecture for the CVAE (synthesis action generation).

`models/material_generator.py` contains the architecture for the CVAE (precursor generation).

`model/token_classifier.py` contains the architecture for the NER model. The methods used for loading in a pretrained ELMo model (via Tensorflow) are also provided here.

`model/paragraph_classifier.py` contains the architecture and code used for the paragraph classifier model.

`data/unsynth_recipes_w_citations.json` collects the suggested recipes produced by the CVAE model for screening unsynthesized ABO3-perovskite compounds. The document also contains CVAE-suggested nearest-neighbor literature.

## Citing 📚

If you use this work (e.g., the NER model, the generative models, the pre-trained embeddings), please cite the following work(s) as appropriate:

```
Kim, E., Jensen, Z., Grootel, A.V., Huang, K., Staib, M., Mysore, S., Chang, H.S., Strubell, E., McCallum, A., Jegelka, S. and Olivetti, E., 2020. Inorganic Materials Synthesis Planning with Literature-Trained Neural Networks. Journal of Chemical Information and Modeling.
```
