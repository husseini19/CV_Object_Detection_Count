#! usr/bin/python
# -*- coding:utf-8 -*-
# Project Name: AutoMated Feature Detection and Count Using Computer Vision
# Project Team:	Kamrun Naher Sumi, Karim Alhusaini, Mathai Paul, Matt M. Seraj, and Rajpal Virk

# Start the program
print("Program Starts")

# Import required libraries
import os, time, cv2, numpy as np, imutils, easygui

# Start timer
start = time.time()

# Load data
imageFile = easygui.fileopenbox()
path, fileName = os.path.split(imageFile)
sourceImage = cv2.imread(imageFile)


# Show source image file
cv2.imshow("Source Image", sourceImage)
cv2.waitKey(0)
cv2.destroyWindow("Source Image")



# Resize image based on user input
usr_input = input('Resize Image (y for yes or any other key for no): ')
if usr_input == 'y':
    scale = float(input('Scale to resize: '))
    resizedSourceImage = imutils.resize(sourceImage, width=int(sourceImage.shape[1]*scale))
    image = resizedSourceImage.copy()
    print('resized image dimensions: ', resizedSourceImage.shape[:2])
    cv2.imshow("Resized Image", resizedSourceImage) 
    cv2.waitKey(0)
    cv2.destroyWindow("Resized Image")
else:
    pass
    resizedSourceImage = sourceImage.copy()


# Image Warping
## User input to confirm whether warping is required or not?
warp = input("Image warping required (press y for yes or any other key for no): ")
if warp=='y':
	print("Select 4 points on source image making a 'Z' pattern")
	print("Press r to reset if wrong points are selected else press s to save")
	
	### Function to Warp image
	pntList = []
	cloneImage = resizedSourceImage.copy()
	def warp_points(event, x, y, flags, param):
		### Global variables
		global pntList, cloneImage

		### Record Mouse Input
		if event == cv2.EVENT_LBUTTONDOWN:
			cv2.circle(resizedSourceImage, (x, y), 5, (0,0,255), cv2.FILLED)
			pntList.append((x, y))
	while True:
		cv2.imshow("Resized Image to Warp", resizedSourceImage)
		cv2.setMouseCallback("Resized Image to Warp", warp_points)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('r'): #### press r to reset
			resizedSourceImage = cloneImage.copy()
			pntList = []
		elif key == ord('s'):
			break
	width, height = 630, 630
	pts1 = np.float32([[pntList[0]], [pntList[1]], [pntList[2]], [pntList[3]]])
	pts2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
	matrix = cv2.getPerspectiveTransform(pts1, pts2)
	warpImage = cv2.warpPerspective(resizedSourceImage, matrix, (width, height))
	print("Warped Image dimensions are: ", warpImage.shape[:2])
	cv2.destroyWindow("Resized Image to Warp")
	cv2.imshow("Warped Image", warpImage)
	resizedSourceImage = warpImage.copy()
	cv2.waitKey(0)
	cv2.destroyWindow("Warped Image")
	img = warpImage.copy()
elif warp!='y':
	print("No Image Warpping required")
	img = resizedSourceImage.copy()


# Adding padding
ht, wd, cc = img.shape
#print('Height, Width and Colour Channels of source image: ', ht, wd, cc)

hypotenuse = int((ht**2 + wd**2)**0.5)

# create new image of desired size and color for padding
ww = hypotenuse
hh = hypotenuse
color = (255,255,255)
pad_image = np.full((hh,ww,cc), color, dtype=np.uint8)

# compute padded width or height
xx = (ww - wd) // 2
yy = (hh - ht) // 2

# copy cloned image (variable name, image) into center of padding image
pad_image[yy:yy+ht, xx:xx+wd] = img

cv2.imshow("Padded Image", pad_image) 
cv2.waitKey(0)
cv2.destroyWindow("Padded Image")


# Add Blur to Image
image = cv2.bilateralFilter(pad_image, 11, 17,17)

# Convert to Grayscale Image
org_image = image.copy()
backup_org_image = pad_image.copy()
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image_gray.copy()

# Template Selection
crop = False
coordinates = []
## Function to draw rectangle around template (region of interest)
def selectTemplate(event, x, y, flags, param):
	global crop, coordinates, image
	### Record Mouse Input
	if event == cv2.EVENT_LBUTTONDOWN:
		coordinates = [(x, y)]
		crop = True
	elif event == cv2.EVENT_LBUTTONUP:
		coordinates.append((x, y))
		crop = False
		#### Draw rectangle around template
		cv2.rectangle(image, coordinates[0], coordinates[1], (0, 0, 255), 2)

## If there is an existing template available
if os.path.isfile("Template.jpg"):
	print("Template File exists.")
	templateFile = "Template.jpg"
	template = cv2.imread(templateFile, 0)
	#template = cv2.bilateralFilter(template, 11, 17,17)
	cv2.imshow("Existing Template", template)
	cv2.waitKey(0)
	### User input whether to use existing template
	decision = input("Do you want to use existing Template (y for yes or any other key for no): ")

	### User input to use existing template
	if decision=='y':
		print("Using existing template")
	### User input to not use existing template
	else:
		cv2.destroyWindow("Existing Template")
		cloneImage = image.copy()
		#### Select Template
		cv2.namedWindow("Select Template")
		cv2.setMouseCallback("Select Template", selectTemplate)
		while True:
			cv2.imshow("Select Template", image)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("r"):
				image = cloneImage.copy()
			elif key == ord("s"):
				break
		if len(coordinates) == 2:
			template = cloneImage[coordinates[0][1] : coordinates[1][1], coordinates[0][0] : coordinates[1][0]]
			#template = cv2.bilateralFilter(template, 11, 17,17)         
			cv2.imshow("New Template", template)
			cv2.waitKey(0)
			cv2.destroyWindow("New Template")
			cv2.imwrite("Template.jpg", template)
else:
	cloneImage = image.copy()
	### Select Template
	cv2.namedWindow("Select Template")
	cv2.setMouseCallback("Select Template", selectTemplate)
	while True:
		cv2.imshow("Select Template", image)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("r"):
			image = cloneImage.copy()
		elif key == ord("s"):
			break
	if len(coordinates) == 2:
		template = cloneImage[coordinates[0][1] : coordinates[1][1], coordinates[0][0] : coordinates[1][0]]
		#template = cv2.bilateralFilter(template, 11, 17,17)
		cv2.imshow("New Template", template)
		cv2.waitKey(0)
		cv2.destroyWindow("Select Template")
		cv2.imwrite("Template.jpg", template)
	

# Extracting height and width of template
(tH, tW) = template.shape[:2]

# Image Processing - Scaling, Rotation and Template Matching
## Scaling input parameters
decision= input('Do you wish to scale the image (y for yes or any other key for no) : ')
if decision == 'y':
	r1 = float(input("Enter value of smallest scale: "))
	s1 = float(input("Enter value of largest scale: "))
	t1 = int(input("Enter value of scale interval: "))
else:
	r1, s1, t1 = 1, 1, 1

## Rotation angles input parameters
decision= input('Do you wish to rotate the image (y for yes or any other key for no) : ')
if decision == 'y':
	a = int(input("Enter value of ending rotation angle: "))
	b = int(input("Enter value of angle interval: "))
else:
	a, b = 360, 360

## Threshold input
threshold = float(input('Enter the threshold value: '))

## Saving a backup copy of image data before final processing
img_backup = image.copy()


# Function to executed template matching process
def tm(image, template, r1, s1, t1, a, b, threshold ,org_image, backup_org_image, img_backup):

	# Function to get the rotation points 
	def rotate2originalpos(point, origin, degrees):
		radians = np.deg2rad(degrees)#numpy only works for radian
		x,y = point
		new_origin_x, new_origin_y = origin
		new_x = (x - new_origin_x)
		new_y = (y - new_origin_y)
		cos_rad = np.cos(radians)
		sin_rad = np.sin(radians)
		Wo_rotation_x = new_origin_x + cos_rad * new_x + sin_rad * new_y
		wo_rotation_y = new_origin_y + -sin_rad * new_x + cos_rad * new_y
		return Wo_rotation_x,  wo_rotation_y


	rectangles1 = []
	for scale in np.linspace(r1,s1,t1)[::-1]:#(linear parameter: start, stop)
		resized = imutils.resize(image, width = int(image.shape[1] * scale))
		r = image.shape[1] / float(resized.shape[1])
		w,h = resized.shape[:: -1]     
		xoc=(w//2)
		yoc=(h//2) 
		if resized.shape[0] < template.shape[0] or resized.shape[1] < template.shape[1]:
			print("The source image is smaller than template, I quit")
			break

		for angle in np.arange(0,a,b):
			rotated=imutils.rotate(resized, angle)
			res=cv2.matchTemplate(rotated,template, cv2.TM_CCOEFF_NORMED)
			(minVal, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
                    
			threshold = threshold   
			loc = np.where(res>=threshold) 
			if loc: 
				for pt in list(zip(*loc[::-1])):
					x1=(pt[0])
					y1=(pt[1])
					x3=(pt[0]+tW)
					y3=(pt[1]+tH)
					[x1,y1]=rotate2originalpos([x1,y1],[xoc,yoc],-angle) #Rotate point from scaled and rotated image to source image
					[x3,y3]=rotate2originalpos([x3,y3],[xoc,yoc],-angle)
					rect1 = [int(x1*r),int(y1*r), int((x3)*r),int((y3)*r)]
					rectangles1.append(rect1)
                
					# second rectange added to overcome single rectangle removal issue during grouping
					rectangles1.append(rect1)
             
                
	# Grouping rectangles to remove multiple occurances of rectangles drawn during template matching.
	rectangles11, weights = cv2.groupRectangles(rectangles1, groupThreshold=1, eps=0.05)


	# Adding Bounding box and counting rectangles.
	count = 0
	for i in range(len(rectangles11)):
		x1=rectangles11[i][0]
		y1=rectangles11[i][1]
		x3=rectangles11[i][2]
		y3=rectangles11[i][3]
		center_x = x1 + int((x3-x1)/2.0)
		center_y = y1 + int((y3-y1)/2.0)
		top_left=(int(x1),int(y1))
		bottom_right=(int(x3),int(y3))
		#image=cv2.drawMarker(pad_image, (center_x, center_y), color=(0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=40, thickness=6)           
		image=cv2.rectangle(pad_image, top_left, bottom_right, (0, 0,255), 3)
		count +=1
	
	
	cv2.putText(pad_image, 'Detected: ' +str(count), (30,30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 1)
	print("Total Count is: ", count)


	# Save output file
	output_directory = "./Output"
	output_filename = ("Output Image-" + str(r1) + str(s1) + str(t1) + "-" + str(0) + str(a) +str(b) + "-" + str(threshold) + "-" + str(fileName))
	output_file = os.path.join(output_directory, output_filename)
	print(output_file)
	cv2.imwrite(output_file, pad_image)


	# End timer
	end = time.time()
	print("Duration in minutes: ", round(((end-start)/60),2))
	cv2.imshow("Output Image", pad_image)
	cv2.waitKey(0)
	cv2.destroyWindow("Output Image")
	
	
# Loop added for threshold.
while threshold!=0:
	tm(image, template, r1, s1, t1, a, b, threshold ,org_image, backup_org_image, img_backup)
	
	## Threshold input
	## User must input 0 to terminate the loop
	threshold = float(input('Enter the threshold value: '))
	pad_image = backup_org_image.copy()
	image = img_backup.copy()

else:
	print("Threshold is equal to 0.")


# Destroy all windows and end the program
cv2.destroyAllWindows()
print("Program Ends")
