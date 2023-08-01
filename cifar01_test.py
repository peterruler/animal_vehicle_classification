# evaluate the deep model on the test dataset
from keras.datasets import cifar10
from keras.models import load_model
from keras.utils import to_categorical
from keras.optimizers.legacy import SGD
from tensorflow.keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

label_names = ['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	
    # load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	
    # load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")

	# compile model
	opt = SGD(learning_rate=0.001, momentum=0.9)
	loaded_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	# evaluate model on test dataset
	_, acc = loaded_model.evaluate(testX, testY)
	print('> %.3f' % (acc * 100.0))

	y_prob = loaded_model.predict(testX)
	y_classes = y_prob.argmax(axis=-1)
	fig, ax = plt.subplots(6, 8, subplot_kw=dict(xticks=[], yticks=[]), figsize=(10,8))
	fig.tight_layout(pad=2)

	print(trainX.shape)
	# print(len(range(0,10)))
	for i, axi in enumerate(ax.flat):		
			color = 'black'
			value = -1
			for n in range(0,10) :
				if testY[i][n] == 1 :
					value = n
			# print(value)
			# print(y_classes[i])
			if y_classes[i]!=value: 
					color='r'	
			axi.imshow(testX[i], cmap='gray_r')
			axi.set_title(label_names[y_classes[i]], color=color)
	plt.axis('off')
	plt.show()	
# entry point, run the test harness
run_test_harness()
