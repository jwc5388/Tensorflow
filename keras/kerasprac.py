# # # # # import tensorflow as tf
# # # # # import numpy as np

# # # # # from tensorflow.keras.models import Sequential
# # # # # from tensorflow.keras.layers import Dense


# # # # # x = np.array([1,2,3,4,5,6,7,8,9,10]) 
# # # # # y = np.array([2,4,6,8,10,12,14,16,18,20]) 

# # # # # model = Sequential()
# # # # # model.add(Dense(1, input_dim=1))

# # # # # #Mean Squared Error (MSE), which is a common loss function used for regression tasks. 
# # # # # # It calculates the average squared difference between the predicted values and the actual values. 
# # # # # # The model tries to minimize this error during training.

# # # # # # Optimizers are algorithms that adjust the weights of the neural network to minimize the loss function. 
# # # # # # In this case, Adam (short for Adaptive Moment Estimation) is used.
# # # # # model.compile(loss='mse', optimizer = 'adam')

# # # # # #x is input data, y is target data
# # # # # #epochs = number of times the whole dataset is passed into the network during training
# # # # # model.fit(x,y, epochs=11000)

# # # # # result = model.predict(np.array([30]))

# # # # # print('prediction of 20: ' , result)



# # # # # import sklearn as sk
# # # # # from sklearn.datasets import load_boston
# # # # # import numpy as np
# # # # # from tensorflow.keras.models import Sequential
# # # # # from tensorflow.keras.layers import Dense
# # # # # from sklearn.model_selection import train_test_split

# # # # # #1 data
# # # # # dataset = load_boston()
# # # # # #Describe
# # # # # print(dataset.DESCR)
# # # # # print(dataset.feature_names)

# # # # # x = dataset.data
# # # # # y = dataset.target

# # # # import sklearn as sk

# # # # from sklearn.datasets import load_boston
# # # # import numpy as np
# # # # from tensorflow.keras.models import Sequential
# # # # from tensorflow.keras.layers import Dense
# # # # from sklearn.model_selection import train_test_split

# # # # dataset = load_boston()
# # # # #Describe 
# # # # print(dataset.DESCR)
# # # # print(dataset.feature_names)

# # # # x = dataset.data
# # # # y = dataset.target

# # # # print(x.shape)

# # # # x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state=3)


# # # # #2 model
# # # # model = Sequential()
# # # # model.add(Dense(10, input_dim = 13))
# # # # model.add(Dense(100))
# # # # model.add(Dense(100))
# # # # model.add(Dense(100))
# # # # model.add(Dense(100))
# # # # model.add(Dense(100))
# # # # model.add(Dense(1))


# # # # #3 compile and train
# # # # model.compile(loss = 'mse', optimizer = 'adam')
# # # # model.fit(x_train, y_train, epochs = 200, batch_size = 1)

# # # # loss = model.evaluate(x_test,y_test)
# # # # result = model.predict(x_test)

# # # # from sklearn.metrics import r2_score

# # # # r2 = r2_score(y_test, result)
# # # # print("r2 score:", r2)

# # # #열= column= 속성 =FEATURE

# # # # import numpy as np
# # # # from tensorflow.keras.models import Sequential
# # # # from tensorflow.keras.layers import Dense
# # # # from sklearn.model_selection import train_test_split


# # # # #1 data

# # # # x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
# # # # y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])

# # # # #2 model
# # # # x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=3)

# # # # model  = Sequential()
# # # # model.add(Dense(10, input_dim = 1))
# # # # model.add(Dense(100))
# # # # model.add(Dense(100))
# # # # model.add(Dense(100))
# # # # model.add(Dense(100))
# # # # model.add(Dense(10))
# # # # model.add(Dense(1))


# # # # #3 compile and train
# # # # model.compile(loss = 'mse', optimizer = 'adam')

# # # # model.fit(x_train, y_train, epochs = 500, batch_size = 1)

# # # # #4 evaluate and predict

# # # # loss = model.evaluate(x_test, y_test)

# # # # result = model.predict([x_test])

# # # # from sklearn.metrics import mean_squared_error

# # # # def RMSE(y_test, y_predict):
# # # #     return np.sqrt(mean_squared_error(y_test, y_predict))

# # # # rmse = RMSE(y_test, result)
# # # # print("RMSE:", rmse)

# # # import sklearn as sk
# # # from sklearn.datasets import load_boston
# # # import numpy as np
# # # from tensorflow.keras.models import Sequential
# # # from tensorflow.keras.layers import Dense

# # # from sklearn.model_selection import train_test_split

# # # dataset = load_boston()

# # # x = dataset.data
# # # y = dataset.target


# # # x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=3)



# # # #model

# # # model = Sequential()

# # # model.add(Dense(50, input_dim = 13))
# # # model.add(Dense(100))
# # # model.add(Dense(100))
# # # model.add(Dense(100))
# # # model.add(Dense(100))
# # # model.add(Dense(100))
# # # model.add(Dense(100))
# # # model.add(Dense(1))


# # # model.compile(loss = 'mse', optimizer = 'adam')
# # # model.fit(x_train, y_train, epochs = 100, batch_size = 1)

# # # result = model.predict(x_test)

# # # from sklearn.metrics import r2_score

# # # r2 = r2_score(y_test, result)
# # # print("r2 score:", r2)


# # import numpy as np
# # import pandas as pd

# # from tensorflow.keras.models import Sequential
# # from tensorflow.python.keras.layers import Dense

# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import r2_score, mean_squared_error
# # # path = './_data/dacon/따릉이/'
# # path = './_data/dacon/따릉이/'

# # train_csv = pd.read_csv(path+'train.csv', index_col=0)
# # print(train_csv)

# # test_csv = pd.read_csv(path+ 'test.csv', index_col=0)
# # print(test_csv)

# # test_csv = pd.read_csv(path+'test.csv', index_col=0)



# # submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
# # print(submission_csv)


# # print(train_csv.columns)
# # print(train_csv.info())
# # print(train_csv.describe())



# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import numpy as np

# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

# model = Sequential()
# model.add(Dense(100, input_dim = 1))
# model.add(Dense(200))
# model.add(Dense(200))
# model.add(Dense(200))
# model.add(Dense(1))

# model.compile(loss = 'mse', optimizer = 'adam')
# model.fit(x_train, y_train, epochs = 100, batch_size = 1, verbose = 0)

# #0 = 침묵, 빨리 넘기기
# #1 = default 
# #2 = progress bar delete, 간결해짐
# #3 = epochs 만 나옴. epoch 만 확인하고 싶으면 0,1,2, 이외의 숫자 입력

# loss = model.evaluate(x_test, y_test)
# result = model.predict(np.array([11]))

# print("loss:", loss)
# print("[11]의 예측값:", result)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time

x_train = np.array(range(100))
y_train = np.array(range(100))

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])


model = Sequential()
model.add(Dense(100, input_dim =1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time() #현재 시간을 반환, 시작시간
timestamp = 1747971378.293003 
print(time.ctime(timestamp))
print(start_time)   #1747971378.293003 

model.fit(x_train, y_train, epochs =500, batch_size = 2, verbose = 0)


#0 침묵, 빨리 넘기기
#1 default
#2 프로그래스바 삭제, 간결해짐
#3 에포만 나옴, epoch 만 확인하고 싶으면 0,1,2 이외의 숫자 입력

end_time = time.time()
print('걸린시간:', end_time - star)