# Named Entity Recognition (NER) Using BiLSTM + CRF
This is a application of named entity recognition by using the model structure bidirectional LSTMs and followed by the CRF layer. \
It does not contain the trained model and the data, since for different tasks the model needs different fields of corpus.

### Requirements
python-version=3.7\
packages:
```bash
pip install -r requirements.txt
sudo pip install git+https://www.github.com/keras-team/keras-contrib.git
```

### Data
Drop your training data in ./data/train_data.txt. \
The format is:
```text
中 B_institute
国 I_institute
银 I_institute
行 E_institute
```
### Custom settings
modify the values in setting.py
```python
EMBEDDING_DIMENSION = 300
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
DROPOUT = 0.5
RECURRENT_DROPOUT = 0.5
OPTIMIZER = 'adam'
ES_PATIENCE = 10
RL_FACTOR = 0.8
RL_PATIENCE = 2
```

### Training
run train.py

### Prediction & Flask service
run service.py
