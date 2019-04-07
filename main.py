from machine_learning import machine_learning
from imutils import face_utils

import numpy
import imutils
import dlib
import cv2


number_of_attributes = 12


def main():
	numpy.set_printoptions(precision=4)
	while True:
		choice = input("1) Create dataset\n2) Train Models\n3) Test\n4) Exit\nWhat do you want: ")
		if choice == "1":
			machine_learning.createDataset()
		elif choice == "2":
			machine_learning.trainModels(number_of_attributes)
		elif choice == "3":
			machine_learning.test()
		elif choice == "4":
			exit(0)

		print("\n\n\n")


def emotionDetector(img):
	detector = dlib.get_frontal_face_detector()

	# load the input image, resize it, and convert it to gray scale
	image = cv2.imread(img)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the gray scale image
	rectangles = detector(gray, 1)
	# noinspection PyArgumentList
	predictor = dlib.shape_predictor("helpers\\shape_predictor_68_face_landmarks.dat")

	data = list()
	# loop over the face detections
	for (i, rect) in enumerate(rectangles):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		data.append(getFeatures(shape))

	return data


# noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def getFeatures(shape):
	data = list()

	jaw = shape[:17]
	right_eyebrow = shape[17:22]
	left_eyebrow = shape[22:27]
	nose = shape[27:35]
	right_eye = shape[36:42]
	left_eye = shape[42:48]
	mouth = shape[48:68]

	jaw_max_x, jaw_max_y = jaw.max(axis=0)
	jaw_min_x, jaw_min_y = jaw.min(axis=0)
	jaw_height = jaw_max_y - jaw_min_y
	jaw_width = jaw_max_x - jaw_min_x

	right_eyebrow_max_x, right_eyebrow_max_y = right_eyebrow.max(axis=0)
	right_eyebrow_min_x, right_eyebrow_min_y = right_eyebrow.min(axis=0)

	left_eyebrow_max_x, left_eyebrow_max_y = left_eyebrow.max(axis=0)
	left_eyebrow_min_x, left_eyebrow_min_y = left_eyebrow.min(axis=0)
	left_eyebrow_height = left_eyebrow_max_y - left_eyebrow_min_y
	left_eyebrow_width = left_eyebrow_max_x - left_eyebrow_min_x

	right_eye_max_x, right_eye_max_y = right_eye.max(axis=0)
	right_eye_min_x, right_eye_min_y = right_eye.min(axis=0)
	right_eye_height = right_eye_max_y - right_eye_min_y
	right_eye_width = right_eye_max_x - right_eye_min_x

	left_eye_max_x, left_eye_max_y = left_eye.max(axis=0)
	left_eye_min_x, left_eye_min_y = left_eye.min(axis=0)
	left_eye_height = left_eye_max_y - left_eye_min_y
	left_eye_width = left_eye_max_x - left_eye_min_x

	nose_max_x, nose_max_y = nose.max(axis=0)
	nose_min_x, nose_min_y = nose.min(axis=0)
	nose_height = nose_max_y - nose_min_y
	nose_width = nose_max_x - nose_min_x

	mouth_max_x, mouth_max_y = mouth.max(axis=0)
	mouth_min_x, mouth_min_y = mouth.min(axis=0)
	mouth_height = mouth_max_y - mouth_min_y
	mouth_width = mouth_max_x - mouth_min_x

	rate_jaw_mouth_height = mouth_height / jaw_height
	rate_jaw_mouth_width = mouth_width / jaw_width
	rate_jaw_nose_width = nose_width / jaw_width
	rate_eyebrow = left_eyebrow_height / left_eyebrow_width
	rate_eyebrow_eye_distance = (left_eye_min_y - left_eyebrow_min_y) / left_eye_width
	rate_eye_left = left_eye_height / left_eye_width
	rate_eye_right = right_eye_height / right_eye_width
	rate_eye_height_1eft = left_eye_height / (left_eye_max_y - left_eyebrow_min_y)
	rate_eye_height_right = right_eye_height / (right_eye_max_y - right_eyebrow_min_y)
	rate_mouth = mouth_height / mouth_width
	rate_mouth_nose_distance_1 = (mouth_max_y - nose_max_y) / mouth_height
	rate_mouth_nose_distance_2 = (mouth_min_y - nose_max_y) / mouth_height

	data.append(rate_jaw_mouth_height)
	data.append(rate_jaw_mouth_width)
	data.append(rate_jaw_nose_width)
	data.append(rate_eyebrow)
	data.append(rate_eyebrow_eye_distance)
	data.append(rate_eye_left)
	data.append(rate_eye_right)
	data.append(rate_eye_height_1eft)
	data.append(rate_eye_height_right)
	data.append(rate_mouth)
	data.append(rate_mouth_nose_distance_1)
	data.append(rate_mouth_nose_distance_2)

	return data


if __name__ == "__main__":
	main()
