#!/usr/bin/env bash

# Download QuAC
mkdir -p QuAC_data
wget https://s3.amazonaws.com/my89public/quac/train.json -O QuAC_data/train.json
wget https://s3.amazonaws.com/my89public/quac/val.json -O QuAC_data/dev.json

# Download CoQA
mkdir -p CoQA
wget https://worksheets.codalab.org/rest/bundles/0xe3674fd34560425786f97541ec91aeb8/contents/blob/ -O CoQA/train.json
wget https://worksheets.codalab.org/rest/bundles/0xe254829ab81946198433c4da847fb485/contents/blob/ -O CoQA/dev.json

# Download GloVe
mkdir -p glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove/glove.840B.300d.zip
unzip glove/glove.840B.300d.zip -d glove

# Download CoVe
wget https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-b142a7f2.pth -O glove/MT-LSTM.pth

# Download SpaCy English language models
python -m spacy download en
