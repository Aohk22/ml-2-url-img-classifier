Workflow:

1. Get data
2. Partition (test/training)
3. Tokenize + extract lexical features
4. Train Word2Vec (using tokens)
5. Vectorize (vector from Word2Vec + lexical features)
6. Train model (Linear SVM)
7. Get classification report

Detailed tokenizing and feature extraction in [feature_extract.py](./feature_extract.py)  
Description in [procedure.md](../procedure.md)

