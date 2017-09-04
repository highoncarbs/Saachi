# Recognition for Politicians

import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import time
import matplotlib
matplotlib.pyplot.ion()

classes = ["", "Narendra Modi" , "Rahul Gandi"]

def converttoRGB(image):
	return cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

def findFaces(image, scaleFactor=1.2):
	gray_img = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
	# print gray_img
	face_cascade = cv2.CascadeClassifier('/home/padam/Documents/git/Saachi/data/lbpcascade_frontalface.xml')
	# print haar_face_cascade
	face = face_cascade.detectMultiScale(gray_img , scaleFactor= 1.2 , minNeighbors=5)
	
	if(len(face)==0):
	#! Imp -> Check for No Images
		return None , None
	(x,y,w,h) = face[0]
	# Return only detected face 
	return gray_img[y:y+w , x:x+h] ,face[0] 

def prepare_data(data_path):
	dirs = os.listdir(data_path)
	faces = []
	labels = []

	for dir_name in dirs:
		subject_data_path = data_path+"/"+dir_name
		subject_images = os.listdir(subject_data_path)
		label = int(dir_name.replace("s", ""))
		for image_name in subject_images:
			image_path = subject_data_path+"/"+image_name
			# print image_path
			image_temp = cv2.imread(image_path)
			# print image_temp
			plt.show(image_temp)
			plt.pause(0.1)

			# detect face 
			face , rect = findFaces(image_temp)

			if face is not None:
				faces.append(face)
				labels.append(label)
	return faces , labels

print("Prepping data..")
faces , labels = prepare_data("/home/padam/Documents/git/Saachi/data/train")
print("Total faces: " , len(faces))
print("Total labels: ", len(labels))


# Using LBPH classifier 
# ** Set API for all classifiers -> Haar Cascade , FisherFace ?

face_algo = cv2.face.LBPHFaceRecognizer_create()
face_algo.train(faces , np.array(labels))

def drw_rectangle(img , rect):
	(x,y,w,h) = rect
	cv2.rectangle(img,(x,y),(x+w , y+h) , (0,255,0),2)

def drw_text(img , text , x,y):
	cv2.putText(img , text,(x,y) , cv2.FONT_HERSHEY_PLAIN , 1.5 , (0,255,0),2)

def predict_face(test_image):
	image_temp = test_image.copy()
	face , rect = findFaces(image_temp)

	label = face_algo.predict(face)
	print label
	label_text = classes[label[0]]
	drw_rectangle(image_temp ,rect)
	drw_text(image_temp , label_text , rect[0] , rect[1]-5)
	return image_temp

print("let it predict...")

base_tests = "/home/padam/Documents/git/Saachi/data/val/"
test_img1 = cv2.imread(base_tests+"testmodi.jpg")
test_img2 = cv2.imread(base_tests+"testrahul.jpeg")
test_img3 = cv2.imread(base_tests+"testkejriwal.jpeg")

pre_1 = predict_face(test_img1)
pre_2 = predict_face(test_img2)
pre_3 = predict_face(test_img3)

print("Prediction complete biatch")
plt.show( pre1)
plt.pause(0.5)
plt.show(pre2)
plt.pause(0.5)
plt.show(pre3)
plt.pause(0.5)