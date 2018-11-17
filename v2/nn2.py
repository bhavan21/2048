from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy


def defineModel():
	model = Sequential()
	model.add(Dense(100, input_dim=192, activation='relu'))
	model.add(Dense(20, activation='relu'))
	model.add(Dense(4, activation='linear'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
	return model


def loadModel():
	try:
		json_file = open('model2.json', 'r')
	except:
		print("Creating JSON file 2 and weights file")
		return defineModel()

	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	model.load_weights("model2.h5")

	model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

	return model

def saveModel(model):
	model_json = model.to_json()
	with open("model2.json", "w") as json_file:
	    json_file.write(model_json)
	model.save_weights("model2.h5")
	print("Saved model to disk")

def getQ(model,X):
	return model.predict(numpy.array([X]))[0]

def train(model,X,Y):
	model.fit(numpy.array(X), numpy.array(Y), epochs=5, batch_size=50,verbose=0)



# model = loadModel();
# x  = numpy.array(numpy.array([[0]*768,[1]*768]))
# y = model.predict(x)
# print(y)
# print(y[0][0])