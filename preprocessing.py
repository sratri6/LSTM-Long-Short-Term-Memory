from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical

FEATURES=['SNR Receiver','SNR Stages','BER Receiver','Modulation Depth']
TARGET='Signal Quality'

def preprocess(data):
    X=data[FEATURES].values
    y=data[TARGET].values
    encoder=LabelEncoder()
    y_enc=encoder.fit_transform(y)
    y_hot=to_categorical(y_enc)
    scaler=MinMaxScaler()
    X_scaled=scaler.fit_transform(X)
    return X_scaled,y_hot,encoder,scaler
