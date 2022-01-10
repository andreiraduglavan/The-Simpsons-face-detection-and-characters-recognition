import os
import numpy as np
import cv2 as cv
from collections import defaultdict
from YWO import isYellow

root_path = "../../CAVA-2021-TEMA2/antrenare/"

names  = ["bart", "homer", "lisa", "marge"]

def intersection_over_union(bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)


        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

image_names = []
bboxes = []
characters = []
nb_examples = 0
Q=defaultdict(lambda: defaultdict(lambda: []))

for name in names:
	filename_annotations = root_path + name + ".txt"
	f = open(filename_annotations)
	for line in f:
		a = line.split(os.sep)[-1]
		b = a.split(" ")
		
		image_name = root_path + name + "\\" + b[0]
		bbox = [int(b[1]),int(b[2]),int(b[3]),int(b[4])]
		character = b[5][:-1]
		Q[image_name]['bboxes'].append(bbox)
		Q[image_name]['character'].append(character)
		nb_examples = nb_examples + 1

width_hog = 36
height_hog = 36

try:
	os.mkdir("../data/exempleNegative/")
except:
	pass
scales=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]
#compute negative examples using 36 X 36 template
idx=0
for image_name in Q:
		idx+=1
		print(image_name)
		img = cv.imread(image_name)
		print("img shape")
		print(img.shape)		
		
		for scale in scales:
			img_scaled=cv.resize(img, (0,0), fx=scale, fy=scale)
			num_rows = img_scaled.shape[0]
			num_cols = img_scaled.shape[1]
			
			#genereaza 10 exemple negative fara sa compari cu nimic, iei ferestre la intamplare 36 x 42
			if num_rows>=36 and num_cols>=36:
				for i in range(10):
					
					x = np.random.randint(low=0, high=num_cols - width_hog)
					y = np.random.randint(low=0, high=num_rows - height_hog)
					
					bbox_curent = [x, y, x + width_hog, y + height_hog]
					
					xmin = bbox_curent[0]
					ymin = bbox_curent[1]
					xmax = bbox_curent[2]
					ymax = bbox_curent[3]
					negative_example = img_scaled[ymin:ymax,xmin:xmax]
					yellowValue=np.sum(isYellow(negative_example))
					if yellowValue>260:
						sum=0
						for bbox in Q[image_name]['bboxes']:
							bbox=(np.array(bbox)*scale).astype(int)
							sum+=intersection_over_union(bbox_curent, bbox)
						if sum==0:	
							filename = "../data/exempleNegative/"+ str(idx)+'_'+str(scale)+'_'+str(i) + ".jpg"
							cv.imwrite(filename,negative_example)
		idx+=1