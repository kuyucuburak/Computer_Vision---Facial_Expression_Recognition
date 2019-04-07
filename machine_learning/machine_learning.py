from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.svm import SVC
from matplotlib import pyplot, gridspec
from main import emotionDetector
from PIL import Image

import pandas
import numpy
import glob
import io
import os


model_happy = joblib.load('machine_learning\\model_happy.pkl')
model_neutral = joblib.load('machine_learning\\model_neutral.pkl')
model_sad = joblib.load('machine_learning\\model_sad.pkl')
model_surprised = joblib.load('machine_learning\\model_surprised.pkl')
model_sleepy = joblib.load('machine_learning\\model_sleepy.pkl')


def createDataset():
	def writeData(array):
		numpy.savetxt("temp.txt", array, delimiter=', ', fmt='%.4f')

		files = ['machine_learning\\dataset.txt', 'temp.txt']
		lines = io.StringIO()
		for file_dir in files:
			with open(file_dir, 'r') as file:
				lines.write(file.read())

		lines.seek(0)
		file = open("machine_learning\\dataset.txt", "w")
		file.write(lines.read())
		file.close()

		os.remove("temp.txt")

	def getData(path, value):
		print("-> ", path)
		array = []
		for file in glob.glob(path):
			features = emotionDetector(file)
			features = numpy.array(features)
			features = numpy.append(features, value)
			array.append(features)
		array = numpy.asarray(array)
		writeData(array)

	try:
		os.remove("machine_learning\\dataset.txt")
	except OSError:
		pass

	try:
		datasetFile = "machine_learning\\dataset.txt"
		open(datasetFile, 'x')
	except FileExistsError:
		pass

	print("\nI am started to creating dataset!")
	getData("database\\happy\\*.*", 1)
	getData("database\\neutral\\*.*", 2)
	getData("database\\sad\\*.*", 3)
	getData("database\\sleepy\\*.*", 4)
	getData("database\\surprised\\*.*", 5)


def trainModels(number_of_attributes):
	def trainModel(index, value):
		data = numpy.loadtxt("machine_learning\\dataset.txt", delimiter=",")
		data[:, index][data[:, index] != value] = 0
		data[:, index][data[:, index] == value] = 1
		numpy.random.seed(50)
		numpy.random.shuffle(data)

		results = data[:, index:index + 1]
		data = data[:, 0:index]

		parameters = {'gamma': (0.1, 0.01, 0.001), 'C': [1, 10, 100], 'degree': [3, 4, 5]}
		svc = SVC(probability=True)
		svc = GridSearchCV(svc, parameters, cv=2)
		svc.fit(data, numpy.array(results).ravel())

		return svc

	global model_happy, model_neutral, model_sad, model_sleepy, model_surprised

	print("\nI am started to train model!")

	print("-> happy")
	model_happy = trainModel(number_of_attributes, 1)
	joblib.dump(model_happy, 'machine_learning\\model_happy.pkl')

	print("-> neutral")
	model_neutral = trainModel(number_of_attributes, 2)
	joblib.dump(model_neutral, 'machine_learning\\model_neutral.pkl')

	print("-> sad")
	model_sad = trainModel(number_of_attributes, 3)
	joblib.dump(model_sad, 'machine_learning\\model_sad.pkl')

	print("-> sleepy")
	model_sleepy = trainModel(number_of_attributes, 4)
	joblib.dump(model_sleepy, 'machine_learning\\model_sleepy.pkl')

	print("-> surprised")
	model_surprised = trainModel(number_of_attributes, 5)
	joblib.dump(model_surprised, 'machine_learning\\model_surprised.pkl')


def test():
	def testImage():
		score_happy = model_happy.predict_proba(emotionDetector(path))[:, 1]
		score_neutral = model_neutral.predict_proba(emotionDetector(path))[:, 1]
		score_sad = model_sad.predict_proba(emotionDetector(path))[:, 1]
		score_surprised = model_surprised.predict_proba(emotionDetector(path))[:, 1]
		score_sleepy = model_sleepy.predict_proba(emotionDetector(path))[:, 1]

		if score_happy > score_neutral and score_happy > score_sad and score_happy > score_sleepy and score_happy > score_surprised:
			pyplot.title("happy")
		elif score_neutral > score_happy and score_neutral > score_sad and score_neutral > score_sleepy and score_neutral > score_surprised:
			pyplot.title("neutral")
		elif score_sad > score_neutral and score_sad > score_happy and score_sad > score_sleepy and score_sad > score_surprised:
			pyplot.title("sad")
		elif score_sleepy > score_neutral and score_sleepy > score_sad and score_sleepy > score_happy and score_sleepy > score_surprised:
			pyplot.title("sleepy")
		elif score_surprised > score_neutral and score_surprised > score_sad and score_surprised > score_sleepy and score_surprised > score_happy:
			pyplot.title("surprised")

		dataFrame = pandas.DataFrame(({'Happy': score_happy, 'Neutral': score_neutral, 'Sad': score_sad, 'Sleepy': score_sleepy, 'Surprised': score_surprised, }))
		pyplot.plot('x', 'Happy', data=dataFrame, marker='o', markerfacecolor=[1, 0, 0], markersize=6, color=[1, 0, 0], linewidth=2, label='Happy: ' + str(score_happy))
		pyplot.plot('x', 'Neutral', score_neutral, data=dataFrame, marker='o', markerfacecolor=[0, 1, 0], markersize=6, color=[0, 1, 0], linewidth=2, label='Neutral: ' + str(score_neutral))
		pyplot.plot('x', 'Sad', data=dataFrame, marker='o', markerfacecolor=[0, 0, 1], markersize=6, color=[0, 0, 1], linewidth=2, label='Sad: ' + str(score_sad))
		pyplot.plot('x', 'Sleepy', data=dataFrame, marker='o', markerfacecolor=[1, 1, 0], markersize=6, color=[1, 1, 0], linewidth=2, label='Sleepy' + str(score_sleepy))
		pyplot.plot('x', 'Surprised', data=dataFrame, marker='o', markerfacecolor=[1, 0, 1], markersize=6, color=[1, 0, 1], linewidth=2, label='Surprised: ' + str(score_surprised))

		pyplot.legend(loc=4, prop={'size': 7})

	gs = gridspec.GridSpec(3, 4)
	pyplot.figure()

	print("-> happy")
	path = "test_images\\test1\\t_happy.png"
	pyplot.subplot(gs[0, 0])
	pyplot.imshow(Image.open(path).convert('RGB'))
	pyplot.subplot(gs[0, 1])
	testImage()

	print("-> neutral")
	path = "test_images\\test1\\t_neutral.png"
	pyplot.subplot(gs[0, 2])
	pyplot.imshow(Image.open(path).convert('RGB'))
	pyplot.subplot(gs[0, 3])
	testImage()

	print("-> sad")
	path = "test_images\\test1\\t_sad.png"
	pyplot.subplot(gs[1, 0])
	pyplot.imshow(Image.open(path).convert('RGB'))
	pyplot.subplot(gs[1, 1])
	testImage()

	print("-> sleepy")
	path = "test_images\\test1\\t_sleepy.png"
	pyplot.subplot(gs[1, 2])
	pyplot.imshow(Image.open(path).convert('RGB'))
	pyplot.subplot(gs[1, 3])
	testImage()

	print("-> surprised")
	path = "test_images\\test1\\t_surprised.png"
	pyplot.subplot(gs[2, 0])
	pyplot.imshow(Image.open(path).convert('RGB'))
	pyplot.subplot(gs[2, 1])
	testImage()

	pyplot.show()
