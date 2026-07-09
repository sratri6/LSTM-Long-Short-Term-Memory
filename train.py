from sklearn.model_selection import train_test_split

def train_model(model,X,y,epochs=50,batch_size=32):
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42,shuffle=False)
    history=model.fit(
        X_train,y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    return history,X_test,y_test
