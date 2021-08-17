import cv2
import numpy as np
import random
import os
import time
'''
Blur tests
My idea to divide an image into polygons then set the color of the polygon as
some function of the pixels it contains.

Using squares, this is just like Gaussian blur if we use Gaussian?

I want to see what it looks like with other shapes, probably need to figure out a way
to create a set of polygons
Maybe can use the random triangle algorithm from before
'''
# can consider blurring at the end or something more aggressive to get rid of the lines
random.seed(8162021)
IMAGE_PATH = 'portrait.jpg'
def main():
	# read the image
	image = cv2.imread(IMAGE_PATH)
	rows, cols, channels = image.shape
	#image = cv2.resize(image, (int(cols/5),int(rows/5)))
	print(image.shape)
	numRows = 100
	numCols = 80
	grid = (numRows, numCols)
	pointSet, polygonSet = startSquares(image, numRows, numCols)
	frames = 10

	imageGrid = ImageGrid(image, grid)
	#print(triangleSet)

	# render(image, polygonSet)
	# print(pointSet)
	# #print(polygonSet)
	# #print(image.shape[:2])
	# movePoints(pointSet, polygonSet, image.shape[:2], travel=(20,20))
	# render(image, polygonSet)

	# http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html

	# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	# FPS = 20.0
	# out = cv2.VideoWriter('landscape_mosaic.mp4', fourcc, FPS, (cols, rows))
	for i in range(frames+1):
		#print(polygonSet)
		startTime = time.time()
		savePath = os.path.join(IMAGE_PATH.split('.')[0],(str(i) + '.jpg'))
		#savePath = 'landscape' + str(i) + '.jpg'
		#frame = render(image, polygonSet, savePath)
		frame = render(imageGrid.getImage(), imageGrid.getPolygonSet(), savePath)
		#showImage(frame/255, 'frame')
		cv2.imwrite(savePath, frame)
		#out.write(frame)
		#pointSet, polygonSet = movePoints(pointSet, polygonSet,image.shape[:2], grid,travel=(5,5))
		imageGrid.movePoints()
		print('completed: ', savePath, ' in ', (time.time() - startTime), ' seconds')
	#out.release()
	#cv2.destroyAllWindows()
def render(image, polygonSet, savePath):
	#polygonSet = getPointSet(pointDict)
	rows, cols,c = image.shape
	outImage = np.zeros(image.shape)
	for polygon in polygonSet:
		# get the bounds of the polygon
		bounds = getBounds(polygon)

		top, bottom, left, right = bounds
		zPolygon = zeroPolygon(polygon, bounds)
		# use the bounds to isolate the block of pixels from the image to operate on
		# this should save a lot of space and therefore time
		#print(polygon)
		# create a mask of pixels that fall in this polygon
		#print(polygon)
		#blank = np.zeros(image.shape[:2], dtype='uint8')

		# this should fit the polygon perfectly
		# white if part ofthe polygon, black if not
		blank = np.zeros((bottom-top, right-left), dtype='uint8')
		# print(polygon)
		# print(bounds)
		# print(zPolygon)
		# print([np.array(polygon,dtype=np.int32)])
		# print( [np.array(zPolygon,dtype=np.int32)])
		#print(blank.shape)
		#print(np.array(polygon,dtype=np.int32))
		#mask = cv2.fillPoly(blank, [np.array(polygon,dtype=np.int32)], 255)
		mask = cv2.fillPoly(blank, [np.array(zPolygon,dtype=np.int32)], 255)
		refBlock = image[top:bottom, left:right]
		
		# print(refBlock.shape)
		# print(blank.shape)
		# showImage(refBlock)
		cvMask = cv2.bitwise_and(refBlock, refBlock, mask=mask) #not sure if this is the best way to do this
		#showImage(cvMask)
		#showImage(cv2.resize((cvMask/255), ((int(rows/4), int(cols/4)))))
		# get the pixels from the original image
		npMask = np.ma.masked_values(cvMask,(0,0,0))
		#print(npMask.shape)
		#showImage(cv2.resize((npMask/255), (500,500)))
		
		# determine the color for this polygon
		setColor = np.mean(npMask, axis=(0,1)).astype(np.uint8)
		#print(setColor)
		# c1Mean = np.mean(npMask1)
		# c2Mean = np.mean(npMask2)
		# c3Mean = np.mean(npMask3)

		# set the output to this color

		# block of set color
		colorBlank = np.full(refBlock.shape, setColor)
		# shape the block to match the mask
		colorMask = cv2.bitwise_and(colorBlank, colorBlank, mask=mask)
		# 
		#showImage(colorMask)
		#showImage(colorBlank)
		#print(colorBlank)
		#colorMask = np.ma.masked_where(mask==(255,255,255), colorBlank)
		#colorMask.set_fill_value((0,0,0))
		
		#outImage += colorMask
		# replace pixels intead of adding
		outImage[top:bottom, left:right] += colorMask
		#outImage = np.where(colorMask!=(0,0,0), colorBlank, outImage)
		#showImage(cv2.resize(outImage/255,(500,500)))
		#print(outImage[0][0])
		#showImage(outImage/255)
		#showImage(npMask, 'npMask')
		# get the average value or something else to determine the color of this polygon
		# create the final image
	#print(outImage)
	#showImage(outImage/255, 'render')
	#cv2.imwrite(savePath, outImage)
	return outImage

'''
polygon is col,row
'''
def getBounds(polygon):
	top = None
	bottom = None
	left = None
	right = None
	for point in polygon:
		col, row = point
		if top is None or row < top:
			top = row
		if bottom is None or row > bottom:
			bottom = row
		if left is None or col < left:
			left = col
		if right is None or col > right:
			right = col
	return (top, bottom, left, right)
'''
polygon is col, row
'''
def zeroPolygon(polygon,bounds):
	top, bottom, left, right = bounds
	newPolygon = []
	for point in polygon:
		col, row = point
		newPolygon.append((col-left,row-top))
	return newPolygon
'''
Return a set of points that will be used to create polygons

{point: neighbors?}
'''
def createDotSet(image, dpr, dpc):
	rows, cols, channels = image.shape
	pointSet = set()
	addRow = int(rows/dpr)
	addCol = int(rows/dpc)
	for row in range(dpr + 1):
		for col in range(dpc + 1):
			checkRow = min(row * addRow, rows)
			checkCol = min(col * addCol, cols)
			pointSet.add((checkRow, checkCol))
			if row * addRow >= rows or col * addCol >= cols:
				continue
	return pointSet

'''
Create a set of triagles (p1, p2, p3) given a grid of points

Maybe actually want a dict
dict {point: neighbors}

'''
def startSquares(image, dpr, dpc, triangle=False):
	# find the sqaure, then add the traingles that form the square via the diagonal
	# can realistically do this with any set of quadrilaterals
	rows, cols, channels = image.shape
	#print(image.shape)
	pointSet = set()
	polygonSet = set()
	#pointDict = {}
	addRow = int(rows/dpr) # row stride
	addCol = int(cols/dpc) # col stride
	# print()
	# print('dpr: ', dpr)
	# print('dpc: ', dpc)
	# print('row stride: ', addRow)
	# print('col stride: ', addCol)
	for row in range(1,dpr + 1):
		#print('row: ',row)
		for col in range(1,dpc + 1):
			#print('col: ', col)
			# checkRow = min(row * addRow, rows)
			# checkCol = min(col * addCol, cols)
			# pointSet.add((checkRow, checkCol))

			# get the square assuming row and col and the bottom right corner
			# square = (row, col), (row - addRow, col), (row, col-addCol), (row-addrow, col - addCol)
			# it's possible row and col will overshoot
			# it's possible row - addRow and col- addCol will undershoot

			# bottomRight and topLeft will be shared between the two triangles
			topRow = max(0, (row*addRow) - addRow)
			bottomRow = min(rows-1, row * addRow)

			leftCol = max(0, (col*addCol)-addCol)
			rightCol = min(cols-1, col * addCol)
			
			# bottomRight = (bottomRow, rightCol)
			# topLeft = (topRow, leftCol)
			# bottomLeft = (bottomRow, leftCol)
			# topRight = (topRow, rightCol)

			# reverse these because fillPoly reads the points incorrectly
			bottomRight = (rightCol,bottomRow)
			topLeft = (leftCol,topRow)
			bottomLeft = (leftCol, bottomRow)
			topRight = (rightCol, topRow)
			
			
			for addPoint in ((topLeft, topRight, bottomRight, bottomLeft, topLeft)):
				pointSet.add(addPoint)
			if triangle:

				t1 = (bottomRight, topLeft, bottomLeft, bottomRight) 
				t2 = (bottomRight, topLeft, topRight, bottomRight)
				polygonSet.add(t1)
				polygonSet.add(t2)
				

			else:
				square = (topLeft, topRight, bottomRight, bottomLeft, topLeft)
				polygonSet.add(square)
			#print(square)
			if row * addRow >= rows or col * addCol >= cols:
				continue
	#print(pointSet)
	#print(polygonSet)
	return pointSet, polygonSet
def movePoints(pointSet, polygonSet, imageSize, grid,travel=(2,2)):
	#numPolygons = len(polygonSet)
	#numPoints = len(pointSet)
	updateDict = {}
	newPolygonSet = set()
	#newPointSet = set()
	rows, cols = imageSize
	numRows, numCols = grid
	rowStride = int(rows / numRows)
	colStride = int(cols / numCols)
	print('rowStride: ', rowStride)
	print('colStride: ', colStride)
	for point in pointSet:
		#print('point: ', point)
		# can potentially set bounds here
		# can we reverse engineer the original point? Yes, it should be the closest grid point
		# maybe need to coordinate with the travel distance
		col, row = point
		print('point:',(row, col))
		# restrain boundary points

		#print('row',row, 'col', col)
		#print('check top: ', row - int(row % rowStride))
		top = max(0,row - int((row+1) % int(rowStride/2)) - int(rowStride/2))
		#print('top row: ', top)
		bottom = min(top + rowStride, rows-1) if top != 0 else int(rowStride/2)
		#print('bottom row: ', bottom)

		left = max(0, col - int((col+1) % int(colStride/2)) - int(colStride/2))
		print('left col:' ,left)
		#print(int(col%int(colStride/2)))

		print(col - int((col+1) % int(colStride/2)) - int(colStride/2))
		right = min(left+colStride, cols-1) if left != 0 else int(colStride/2)
		#print('right: ', right)
		
		
		bounds = ((top,left), (bottom, right))
		print('bounds:', bounds)

		updateDict[point] = movePointSquare(point, bounds)#movePoint(point, bounds, travel)
	# for point in updateDict:
	# 	print(point , ':', updateDict[point])
	#print(updateDict)
	for polygon in polygonSet:
		tempPolygon = []
		updated = False
		for point in polygon:
			if point not in updateDict:
				# this is a new polygon we just added
				updated=True
				break
			tempPolygon.append(updateDict[point])
		if updated:
			continue
		# trying to do this in place because not sure about garbage collection
		#polygonSet.remove(polygon)
		#polygonSet.add(tuple(tempPolygon))
		newPolygonSet.add(tuple(tempPolygon))
		#numPolygons -= 1
		#if numPolygons == 0:
		#	break
	return set(updateDict.values()), newPolygonSet

'''
Choose a random point within bounds that has a different row and col than start
Maybe need to check if bounds is only 1 pixel it won't be possible to get a unique row or col
'''
def movePointSquare(start,bounds):
	#print(bounds)
	sCol, sRow = start
	topLeft, bottomRight = bounds
	top,left = topLeft
	bottom,right = bottomRight
	newRow = random.randint(top, bottom)
	newCol = random.randint(left, right)
	while newRow == sRow and newCol == sCol:
		newRow = random.randint(top, bottom)
		newCol = random.randint(left, right)
	return (newCol, newRow) # reverse for fillPoly
# # need the dict to store references to the same polygon
# # polygonset = {set of points: polygon object}
# # this will allow us to update the same polygon each time on of its points changes
# def getPointSet(pointDict):
# 	#pointSet = set()
# 	return set(pointDict.values())
# '''
# list of polygons:
# for polygon:
# 	move points
# 	tell every point conneted we have moved
# 	update in the list
# 	stop when every

# {point: polygons this point is a part of}
# # move every point
# # update the polygons
# # update set of polygons by removing it then adding a new one with the moved point?
# iterate through point set
# '''
# class QuadPoint():
# 	def __init__(self, point, left, right, top, bottom):
# 		self.point = point
# 		self.left = left
# 		self.right = right
# 		self.top = top
# 		self.bottom = bottom
# 	def move(self, newPoint):
# 		self.point = newPoint
# 		# move this point somewhere randomly within it's neighbors
# 		# determine bounds
# class Polygon():
# 	def __init__(self, points):
# 		# points should be a set of points
# 		self.points = set(points)
# 	def getPolygon:
# 		return list(points).append(self.points[0])
# 	def updatePoint(self, point, newPoint):
# 		self.points.remove(point)
# 		self.points.add(newPoint)
# '''
# Move every point in order
# for old point, in order:
# 	get the polygons in this point
# 	update the points of the polygon with the new points
	
# '''
# '''
# Keep track of which points touch other points
# keep track of which points are polygons
# '''
# class QuadGraph():
# 	def __init__(self):
# 		self.pointDict = {}

# 	def addQuadPoint(self, quadPoint):
# 		if quadPoint.point in self.pointDict:


# '''
# pointSet dict: {point : neighbors}
# move a point:
# 	for each neighbor
# 	{neighbor : updated point}

# maybe add bias for staying away from other points or something
# '''
# def movePoints(pointDict, bounds, travel=(2,2)):
# 	startSet = getPointSet(pointDict)
# 	for point in pointDict:
# 		newPoint = movePoint(point, bounds, travel)
# 		# update all the polygons in pointDict
# 		for polygon in pointDict[point]:
# 			if polygon in startSet:
# 				# remove 

# divide image into grid
# 1 point per square
# keep track of which sqaures should be connected
# each time choose a random point within the square

class ImageGrid():
	def __init__(self, image, grid):
		self.image = image
		self.rows, self.cols = grid
		#self.cols = cols
		self.iRows, self.iCols, channels = image.shape
		self.rowStride = int(self.iRows / self.rows)
		self.colStride = int(self.iCols / self.cols)

		# using a list as a hash map, we want to use the indicies to determine the square bounds when choosing a new point
		self.pointSet = []
		self.polygonSet = set()

		# initialize the point set and create the polygon set
		# polygon set can store indicies to lookup into the point set

		# intialize the first row an col
		row0 = []

		for col in range(self.rows+1):
			# not sure if we need to keep the points in bounds
			row0.append((0, min(col*self.colStride, self.iCols-1)))
		self.pointSet.append(row0)
		for row in range(1,self.rows+1):
			# Start with col element since it won't be included in the loop
			tempRow = [(row * self.rowStride,0)]
			top = max(0, (row*self.rowStride) - self.rowStride)
			bottom = min(self.iRows-1, row * self.rowStride)
			for col in range(1,self.cols+1):
				left = max(0, (col*self.colStride)-self.colStride)
				right = min(self.iCols-1, col * self.colStride)
				# print('top: ', top)
				# print('bottom: ', bottom)
				# print('left: ', left)
				# print('right: ', right)
				# print('\n')
				tempRow.append((bottom, right))
				self.polygonSet.add(((row-1, col-1) , (row-1, col), (row, col), (row, col-1), (row-1, col-1)))
			#print('tempRow', tempRow)
			self.pointSet.append(tempRow)

			# square will be formed from
			# ((top,left), (top,right), (bottom, right), (bottom, left), (top, left))
			# (row-1, col-1) , (row-1, col), (row, col), (row, col-1), (row-1, col-1) 
		# print(self.pointSet)
		# print('\n')
		# print(len(self.polygonSet))

	def getPolygonSet(self):
		retSet = set()
		for polygon in self.polygonSet:
			#print(polygon)
			tempPolygon = []
			for point in polygon:
				i,j = point
				r,c = self.pointSet[i][j]
				tempPolygon.append((c,r)) # reverse here for fill poly
			retSet.add(tuple(tempPolygon))
		#print(retSet)
		return retSet
	def movePoints(self):
		for i, row in enumerate(self.pointSet):
			for j, point in enumerate(row):
				top = (i * self.rowStride) - int(self.rowStride/2)
				bottom = top + self.rowStride
				left = (j * self.colStride) - int(self.colStride/2)
				right = left + self.colStride
				# # choose a new point within this point's square
				# top = self.rowStride * i
				# if i == 0 or i == self.rows:
				# 	bottom = top + int(self.rowStride/2)
				# else:
				# 	bottom = top + self.rowStride
				# leftCol = int(self.rowStride/2) + (self.rowStride * (i))
				
				# this needs to be done after determining the intial bounds
				# because bottom and right and dependent on top and left
				top = max(0,top)
				bottom = min(bottom, self.iRows-1)
				left = max(0, left)
				right = min(right, self.iCols-1)
				bounds = ((top,left), (bottom,right))
				if top == bottom or left == right:
					continue
				checkRow, checkCol = checkPoint = self.pointSet[i][j]
				if checkRow < top or checkRow > bottom or checkCol < left or checkCol > right:
					continue
				newPoint = self.chooseNewPoint(self.pointSet[i][j], bounds)
				#newPoint = self.chooseNeighbor(self.pointSet[i][j], bounds)
				self.pointSet[i][j] = newPoint
	def chooseNewPoint(self, start, bounds):
		#print(start)
		sRow, sCol = start
		topLeft, bottomRight = bounds
		top,left = topLeft
		bottom,right = bottomRight

		pRow = list(range(top,bottom+1))
		rowIndex = pRow.index(sRow)
		maxRow = max(len(pRow) - rowIndex-1, rowIndex)
		#maxRow - abs(i-rowIndex)
		rowWeight = [maxRow - abs(i-rowIndex) if i != rowIndex else 0 for i in list(range(len(pRow)))]
		#print(pRow)
		#print(rowWeight)

		pCol = list(range(left,right+1))
		colIndex = pCol.index(sCol)
		maxCol = max(abs(sCol - pCol[0]), abs(sCol - pCol[-1]))
		colWeight = [maxCol - abs(i-colIndex) if i != colIndex else 0 for i in list(range(len(pCol)))]

		# newRow = random.randint(top, bottom)
		# newCol = random.randint(left, right)
		# # print('top: ', top)
		# # print('bottom: ', bottom)
		# # print('left: ', left)
		# # print('right: ', right)
		# # print('\n')
		# while(newRow, newCol) == start:
		# 	newRow = random.randint(top, bottom)
		# 	newCol = random.randint(left, right)
		newRow = random.choices(pRow, rowWeight)[0]
		newCol = random.choices(pCol, colWeight)[0]
		#print((newRow, newCol))
		return (newRow, newCol)
	def chooseNeighbor(self, start, bounds):
		row, col = start

		topLeft, bottomRight = bounds
		top, left = topLeft
		bottom, right = bottomRight
		rowAdd = []
		colAdd = []
		if row > top:
			rowAdd.append(-1)
		if row < bottom:
			rowAdd.append(1)
		if col > left:
			colAdd.append(-1)
		if col < right:
			colAdd.append(1)
		return (row + random.choice(rowAdd), col + random.choice(colAdd))
	# def chooseClosePoint(self, start, bounds, travelRange):
	# 	row, col = start
		
	# 	topLeft, bottomRight = bounds
	# 	top, left = topLeft
	# 	bottom, right = bottomRight

	# 	addRow, addCol = travelRange
		
	# 	rowChoices = []
	# 	for checkRow in range(-addRow, addRow+1):
	# 		rowChoice
	# 	rowChoices = list(range(-addRow, addRow+1)).remove(0)
	# 	colChoices = list(range(-addCol, addCol+1)).remove(0)
		
	# 	rowTransform = random.choice(rowChoices)
	# 	colTransform = random.choice(colChoices)

	# 	return ()

	def getImage(self):
		return self.image
def movePoint(point, bounds, travelDistance):
	maxRow, maxCol = bounds
	col,row = point# reversed
	rowD, colD = travelDistance

	newRow = row
	while newRow == row:
		newRow = row + random.randint(-rowD, rowD)
		newRow = max(0, newRow)
		newRow = min(maxRow-1, newRow)

	newCol = col
	while newCol == col:
		newCol = col + random.randint(-colD, colD)
		newCol = max(0, newCol)
		newCol = min(maxCol-1, newCol)

	return (newCol, newRow) # reversed

def showImage(image,title='image'):
	cv2.imshow(title, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()



'''
set of polygons
polygon is a set of points

to move
for point in set of points:
	move point
	point {old : new}
for polygon:
	new = lookup
'''