import os
import numpy as np
import cv2 as cv
from YWO import isYellow

root_path = "../../CAVA-2021-TEMA2/antrenare/"

names  = ["bart", "homer", "lisa", "marge"]

image_names = []
bboxes = []
characters = []
nb_examples = 0


for name in names:
	filename_annotations = root_path + name+".txt"
	f = open(filename_annotations)
	for line in f:
		a = line.split(os.sep)[-1]
		b = a.split(" ")
		
		image_name = root_path + name + "/" + b[0]
		character = b[5][:-1]
		bbox = [int(b[1]),int(b[2]),int(b[3]),int(b[4]), character]
		
		
		image_names.append(image_name)
		bboxes.append(bbox)
		characters.append(character)
		nb_examples = nb_examples + 1


h=[bbox[3]-bbox[1] for bbox in bboxes if bbox[4]=='homer']
w=[bbox[2]-bbox[0] for bbox in bboxes if bbox[4]=='homer']
h=np.median(h)
w=np.median(w)
print(h,w)
newH=36
newW=36
#compute positive examples using 36 X 36 template
number_roots = 1

try:
	os.mkdir("data/exemplePozitive/")
except:
	pass


for idx, img_name in enumerate(image_names):
	print(idx,img_name)
	img = cv.imread(img_name)
	bbox = bboxes[idx]
	character=characters[idx]
	xmin = bbox[0]
	ymin = bbox[1]
	xmax = bbox[2]
	ymax = bbox[3]
	h=ymax-ymin
	w=xmax-xmin
	print(xmin,ymin,xmax,ymax)
	face = img[ymin:ymax,xmin:xmax]
	face_warped = cv.resize(face,(newW,newH))
	yellowValue=np.sum(isYellow(face_warped))
	if yellowValue>260:
		print("original face shape:",face.shape)
		print("warped face shape:",face_warped.shape)
		filename = "../data/exemplePozitive/"+ str(character)+'_'+str(idx) + ".jpg"
		cv.imwrite(filename,face_warped)