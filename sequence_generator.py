import numpy as np

def create_sequences(X,y,look_back=5):
    X_seq,y_seq=[],[]
    for i in range(len(X)-look_back):
        X_seq.append(X[i:i+look_back])
        y_seq.append(y[i+look_back])
    return np.array(X_seq),np.array(y_seq)
