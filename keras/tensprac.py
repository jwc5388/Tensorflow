import tensorflow as tf
print('version of tf: ', tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Flatten is used to reshape the input data into a 1D vector.

#ReLU stands for Rectified Linear Unit and is a very common activation function. 
#ReLU outputs the input directly if it’s positive, and 0 if it’s negative.
#It helps introduce non-linearity into the model, allowing it to learn more complex patterns.
#So if the output of a neuron is prositive, it passes the value as-is. otherwise it outputs zero

#Dropout layer is a refularizationh technique used during training to prevent overfitting. 
#overfitting happens when the model learns too perform very well on the training data but fails to generalize new, unseen data
#dropout rate 0.2 means that 20% of the neurons in layer will be randomly dropped out during each training step.
# ++ During inference, like making predictions, dropout is turned off, and all neurons are used.
 
#Dense(10) -  this is the output layer of the model. number 10 indicated that there are 10 neurons in this layer,
#corresponding to the 10 possible classes for classification. for example in the MNIST dataset, there are 10 possible digits so you need
#10 neurons to represent the 10 possible outputs
#softmax fuction is used in multi-class classification problems. converts the output of the laer into probabiilities that sum to 1
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)),   #reshapes the input image (28x28) into a 1D vector
        tf.keras.layers.Dense(128, activation='relu'),  #Hidden layer with 128 neurons and ReLU activation
        tf.keras.layers.Dropout(0.2),   #Regularization to prevent overfitting by dropping 20% of neurons during training
        tf.keras.layers.Dense(10, activation='softmax') #output layer with 10 neurons (for 10 classes ) and softmax activation
        
    ]
)

#adam is an adaptive optimizer that adjusts the learning rate based on the progress of the model therefore, you dont need to manually tune it
#Loss sparse categorical crossentropy is a loss function that tells the model how well it is performing after each prediction. \
#it calculates the difference between the predicted output and the true output(target) 
#goal during training is to minimize this loss function
#why sparse? 

#Metrics accuracy - metrics are the criteria used to evaluate the performance of the model during training and testing.
#tell you how well the model is doing based on the evaluation metric you choose

# *** optimizer helps model learn by adjusting weights
# loss measures how good or bad the model is at making predictions
# Metrics like accuracy tell us how well the model is performing

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
predictions = model(x_train[:1]).numpy




