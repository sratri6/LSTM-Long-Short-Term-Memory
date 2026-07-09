from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout

def build_model(input_shape,num_classes):
    model=Sequential([
        LSTM(50,return_sequences=True,input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(num_classes,activation="softmax")
    ])
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    return model
