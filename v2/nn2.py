from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras import optimizers
import numpy


Adam = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


def defineModel():
	model = Sequential()
	model.add(Dense(100, input_dim=192, activation='relu'))
	# model.add(Dense(50, activation='relu'))
	model.add(Dense(40, activation='relu'))
	model.add(Dense(4, activation='linear'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	return model


def loadModel():
	try:
		json_file = open('model3.json', 'r')
	except:
		print("Creating JSON file 2 and weights file")
		return defineModel()

	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	model.load_weights("model3.h5")

	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

	return model

def saveModel(model):
	model_json = model.to_json()
	with open("model3.json", "w") as json_file:
	    json_file.write(model_json)
	model.save_weights("model3.h5")
	# print("Saved model to disk")

def getQ(model,X):
	return model.predict(numpy.array([X]))[0]

def train(model,X,Y):
	model.fit(numpy.array(X), numpy.array(Y), epochs=1, batch_size=32,verbose=0)



# model = loadModel();
# x  = numpy.array(numpy.array([[0]*768,[1]*768]))
# y = model.predict(x)
# print(y)
# print(y[0][0])