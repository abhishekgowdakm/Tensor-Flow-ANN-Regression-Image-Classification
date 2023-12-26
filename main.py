import tensorflow as tf


mnist = tf.keras.datasets.mnist

(X_train,y_train),(X_test,y_test) = mnist.load_data()

X_train,X_test = X_train/255.0,X_test/255.0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
    ]
)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test))

class ANN_Regression():
    def __init__(self,value):
        self.value = value
    
    def predict(self):
        predict = model.predict(self.value)
        predicted_value = tf.argmax(predict,axis=1).numpy()[0]
        return predicted_value

ANN = ANN_Regression(X_test)
prediction_value = ANN.predict()


print(prediction_value)

