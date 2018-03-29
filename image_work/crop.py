#!/usr/bin/python
import os
import cv2

#Adapted from https://github.com/axn337/484_project
wd = os.getcwd()
print("Current directory is \""+wd+"\"")

files = os.listdir(wd)

for name in files:
	debris = name.split('.')
	if (debris[-1] == 'png')&(not debris[0].startswith('reduced_')):
		print("file name :"+name)

		img = cv2.imread(name)
		#print img.shape
		cropped = img[60:420, 80:560]

		#print cropped.shape
		r = 60.0 / cropped.shape[1]
		dim = (60, int(cropped.shape[0] * r))
		 
		# perform the actual resizing of the image and show it
		resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
		#print resized.shape

		cv2.imwrite("reduced_"+name, resized)
