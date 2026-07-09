from data_loader import load_data
from preprocessing import preprocess
from sequence_generator import create_sequences
from model import build_model
from train import train_model
from evaluate import evaluate_model
from utils import plot_history

LOOK_BACK=5

data=load_data()
X,y,encoder,scaler=preprocess(data)
X_seq,y_seq=create_sequences(X,y,LOOK_BACK)
model=build_model((LOOK_BACK,X.shape[1]),y.shape[1])
history,X_test,y_test=train_model(model,X_seq,y_seq)
evaluate_model(model,X_test,y_test)
plot_history(history)
