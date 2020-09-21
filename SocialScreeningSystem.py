# USAGE:
# python SocialScreeningSystem.py -i demo_me.mp4
# python SocialScreeningSystem.py -i demo_me.mp4 -o demo_me_output
# Live Streaming screening
# python SocialScreeningSystem.py  
# python SocialScreeningSystem.py  -o demo_me_output

from identifier import detect_person_class
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
args = vars(ap.parse_args())

dataset_folder="dataset"

# load object class
labelsPath = os.path.sep.join([dataset_folder, "coco.names"])
labelsPath_m = os.path.sep.join([dataset_folder, "classes.txt"])
LABELS = open(labelsPath).read().strip().split("\n")
LABELS_m = open(labelsPath_m).read().strip().split("\n")

# load weights and model configuration
weightsPath = os.path.sep.join([dataset_folder, "yolov3.weights"])
configPath = os.path.sep.join([dataset_folder, "yolov3.cfg"])
weightsPath_m = os.path.sep.join([dataset_folder, "yolov3_training_final.weights"])
configPath_m = os.path.sep.join([dataset_folder, "yolov3_testing.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net_m = cv2.dnn.readNetFromDarknet(configPath_m, weightsPath_m)


ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

print ("Loading...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# loop over the frames from the video stream
while True:
	(grabbed, frame) = vs.read()

	if not grabbed:
		break

	# resize the frame and then detect people (and only people) in it
	frame = imutils.resize(frame, width=700)
	
	###################################################
	# MASK Dedection
	###################################################

	img =  imutils.resize(frame, width=700) # cv2.imread('./testimage.jpg')
	height,width, _ = img.shape
	blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
	net_m.setInput(blob)
	output_layer_names = net.getUnconnectedOutLayersNames()
	layersOutput = net_m.forward(output_layer_names)

	boxes = []
	confidences = []
	class_ids = []

	mask_violation = 0

	for output in layersOutput:
		for detection in output:
			scores  = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
				
				x =  int(center_x - w/2)
				y =  int(center_y - h/2)
				
				boxes.append([x,y,w,h])
				confidences.append((float(confidence)))
				class_ids.append(class_id)
		
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4 )
	font = cv2.FONT_HERSHEY_PLAIN
	
	try:
		for i  in indexes.flatten():
			x,y,w,h = boxes[i]
			label = str(LABELS_m[class_ids[i]])
			confidence = str(round(confidences[i],2))
			color = (0, 255, 0)

			# if the index pair exists within the violation set, then
			# update the color
			if class_ids[i] > 0:
				mask_violation = mask_violation + 1
				color = (0, 0, 255)
			
			cv2.rectangle(img,(x,y),(x+w,y+h), color, 2)
			#cv2.putText(img, label + " " + confidence, (x, y+20), font, 1,(255, 255,255),2)
	except (AttributeError):
		pass
	
	# draw the total number of mask violations on the output frame
	text = "Mask Violations: {}".format(mask_violation)
	cv2.putText(img, text, (10, img.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
	
	###################################################
	# Distance between person Dedection
	###################################################
	results = detect_person_class(frame, net, ln,
		personIdx=LABELS.index("person"))

	violation = set()

	if len(results) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# calibrate the MIN_DISTANCE 50 is used for demo
				# based on the camera settings change this value from 50 to your calibration value
				if D[i, j] < 50:
					violation.add(i)
					violation.add(j)

	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		if i in violation:
			color = (0, 0, 255)

		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	text = "Social Distancing Violations: {}".format(len(violation))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

	# show the output frame
	cv2.imshow("Distancing tracking screen", frame)
	cv2.imshow("Mask detecting screen", img)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	if args["output"] != "" and writer is None:
		# initialize our video writer
		filename =args["output"] + "_distance.avi"
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(filename, fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)
		filename =args["output"] + "_mask.avi"		
		writer2 = cv2.VideoWriter(filename, fourcc, 25,
			(img.shape[1], img.shape[0]), True)

	if writer is not None:
		writer.write(frame)
		writer2.write(img)

		
		