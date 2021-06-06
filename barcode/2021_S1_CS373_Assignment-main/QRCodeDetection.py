
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.path import Path
import matplotlib.patches as patches

import imageIO.png
import math
import cv2


# def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

#     new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
#     return new_array


# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# This method packs together three individual pixel arrays for r, g and b values into a single array that is fit for
# use in matplotlib's imshow method
def prepareRGBImageForImshowFromIndividualArrays(r,g,b,w,h):
    rgbImage = []
    for y in range(h):
        row = []
        for x in range(w):
            triple = []
            triple.append(r[y][x])
            triple.append(g[y][x])
            triple.append(b[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage
    

# This method takes a greyscale pixel array and writes it into a png file
def writeGreyscalePixelArraytoPNG(output_filename, pixel_array, image_width, image_height):
    # now write the pixel array as a greyscale png
    file = open(output_filename, 'wb')  # binary mode is important
    writer = imageIO.png.Writer(image_width, image_height, greyscale=True)
    writer.write(file, pixel_array)
    file.close()

#sum the red green and blue valued pixels to define a greyscale array
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    #initialize the grey scale array
    greyscale_pixel_array = pixel_array_b
    
    for i in range(image_height):
        for o in range(image_width):
            grey = 0.299 * pixel_array_r[i][o] + 0.587 * pixel_array_g[i][o] + 0.114 * pixel_array_b[i][o]
            
            greyscale_pixel_array[i][o] = round(grey)
    return greyscale_pixel_array

#Histogram function
def computeHistogram(pixel_array, image_width, image_height, nr_bins):
    outpt = [0.0 for i in range(nr_bins)]
    for row in range(image_height):
        for column in range(len(pixel_array[row])):
            outpt[int(pixel_array[row][column])] += 1  
    return outpt

#calculate the gradient of adjacent horizontal pixels
def computeHorizontalEdges(pixel_array, image_width, image_height): 
    outpt = []
    for h in range(image_height):
        row = []
        for w in range(image_width):
            if (h == 0) or (h == image_height-1) or (w == 0) or (w == image_width-1):
                val = 0
            else:
                val = ((1/8) * ( pixel_array[h-1][w-1] + 2*pixel_array[h-1][w] +  pixel_array[h-1][w+1]  
                                  - pixel_array[h+1][w-1] - 2*pixel_array[h+1][w] - pixel_array[h+1][w+1])) 
            row.append(float(val))
        outpt.append(row)
    return outpt

#calculate the gradient of adjacent vertical pixels
def computeVerticalEdges(pixel_array, image_width, image_height): 
    outpt = []
    for h in range(image_height):
        row = []
        for w in range(image_width):
            if (h == 0) or (h == image_height-1) or (w == 0) or (w == image_width-1):
                val = 0
            else:
                val = (1/8)* ( -pixel_array[h-1][w-1] - 2*pixel_array[h][w-1] - pixel_array[h+1][w-1] 
                                   + pixel_array[h-1][w+1] + 2*pixel_array[h][w+1] + pixel_array[h+1][w+1])
            row.append(float(val))
        outpt.append(row)
    return outpt

#calculate the edge gradient by comparing vertical and horizontal magnitudes
def computeEdgeMagnitude(hEdges, vEdges, image_width, image_height):
    outpt = []
    for h in range(image_height):
        row = []
        for w in range(image_width):
            
            val = math.sqrt( hEdges[h][w]*hEdges[h][w] + vEdges[h][w]*vEdges[h][w] )
            row.append(float(val))
        outpt.append(row)
    return outpt

#return the most and least intense pixels in the image
def findMinMax(pixel_array, image_width, image_height):
    min = 255
    max = 0
    for i in range(image_height):
        for o in range(image_width):
            if pixel_array[i][o] < min:
                min = pixel_array[i][o]
            if pixel_array[i][o] > max:
                max = pixel_array[i][o]
    return min, max

#apply blur by merging the intensity of surrounding pixels in the neighbourhood
def computeGaussianBlur(pixel_array, image_width, image_height):
    #apply boarder boundary padding to the pixel matrix
    for i in range(len(pixel_array)):
        pixel_array[i].insert(0, pixel_array[i][0])
        pixel_array[i].insert(len(pixel_array[i]), pixel_array[i][-1])
    pixel_array.insert(0, pixel_array[0])
    pixel_array.insert(len(pixel_array), pixel_array[-1])

    #apply blur to image
    outpt = []
    for h in range(1, image_height+1):
        row = []
        for w in range(1,image_width+1):
            val = (1/16) * (pixel_array[h-1][w-1] + 2*pixel_array[h-1][w] + pixel_array[h-1][w+1]
                            + 2*pixel_array[h][w-1] + 4*pixel_array[h][w] + 2*pixel_array[h][w+1] 
                            + pixel_array[h+1][w-1] + 2*pixel_array[h+1][w] + pixel_array[h+1][w+1] )
            row.append(float(val))
        outpt.append(row)

    #re-distribute values between 0 and 255
    min, max = findMinMax(outpt, image_width, image_height)
    for h in range(image_height):
        for w in range(image_width):
            outpt[h][w] = (outpt[h][w] - min)*((255)/(max-min))
    return outpt

#simplify values to either 0 or 255 based on threshold value
def computeThreshold(pixel_array, threshold_value, image_width, image_height):
    for i in range(image_height):
        for o in range(image_width):
            if pixel_array[i][o] < threshold_value:
                pixel_array[i][o] = 0
            else:
                pixel_array[i][o] = 255
    return pixel_array

#morphological closing, performed using a dilation followed by an erosion
def computeMorphologicalClosing(pixel_array, image_width, image_height):
    dilatedPixels = []
    for i in range(image_height):
        dilatedPixels.append([0]*image_width)

    #running dilation on image
    #checking at pixels in a 3x3 neighbourhood around pixel_array[h][w], taking into account edge cases
    for h in range(image_height):
        for w in range(image_width):
            if h == 0:
            #occurs at top row
                if w == 0:
                #occurs at top left corner
                    if (pixel_array[h][w] + pixel_array[h][w+1] + pixel_array[h+1][w] + pixel_array[h+1][w+1]) > 0:
                        dilatedPixels[h][w] = 1
                if w == image_width-1:
                #occurs at top right corner
                    if (pixel_array[h][w] + pixel_array[h][w-1] + pixel_array[h+1][w] + pixel_array[h+1][w-1]) > 0:
                        dilatedPixels[h][w] = 1
                else:
                    if (pixel_array[h][w-1]+pixel_array[h][w]+pixel_array[h][w+1]
                    +pixel_array[h+1][w-1]+pixel_array[h+1][w]+pixel_array[h+1][w+1]) > 0:
                        dilatedPixels[h][w] = 1

            elif h == image_height-1:
            #occurs at bottem row
                if w == 0:
                #occors at bottem left corner
                    if (pixel_array[h][w] + pixel_array[h][w+1] + pixel_array[h-1][w] + pixel_array[h-1][w+1]) > 0:
                        dilatedPixels[h][w] = 1
                if w == image_width-1:
                #occors at bottem right corner
                    if (pixel_array[h][w] + pixel_array[h][w-1] + pixel_array[h-1][w] + pixel_array[h-1][w-1]) > 0:
                        dilatedPixels[h][w] = 1
                else:
                    if (pixel_array[h][w-1]+pixel_array[h][w]+pixel_array[h][w+1]
                    +pixel_array[h-1][w-1]+pixel_array[h-1][w]+pixel_array[h-1][w+1]) > 0:
                        dilatedPixels[h][w] = 1

            elif w == 0:
            #occurs at left-most column
                if (pixel_array[h-1][w]+pixel_array[h-1][w+1]+
                pixel_array[h][w]+pixel_array[h][w+1]+
                pixel_array[h+1][w]+pixel_array[h+1][w+1]) > 0:
                    dilatedPixels[h][w] = 1

            elif w == image_width-1:
            #occurs at left-most column
                if (pixel_array[h-1][w]+pixel_array[h-1][w-1]+
                pixel_array[h][w]+pixel_array[h][w-1]+
                pixel_array[h+1][w]+pixel_array[h+1][w-1]) > 0:
                    dilatedPixels[h][w] = 1

            else:
                #for all non-edge cases
                val = pixel_array[h][w]
                if (pixel_array[h-1][w-1] + pixel_array[h-1][w] + pixel_array[h-1][w+1]
                + pixel_array[h][w-1] + pixel_array[h][w] + pixel_array[h][w+1]
                + pixel_array[h+1][w-1] + pixel_array[h+1][w] + pixel_array[h+1][w+1]) >0:
                    dilatedPixels[h][w] = 1
                else:
                    dilatedPixels[h][w] = 0
    
    #running erosion on the dilated image (edge cases can be ignored)
    outpt = []
    for i in range(image_height):
        outpt.append([0]*image_width)

    for h in range(1, image_height-1):
        for w in range(1, image_width-1):
            
            if dilatedPixels[h][w] != 0:
                val = dilatedPixels[h][w]
                if (dilatedPixels[h-1][w-1] + dilatedPixels[h-1][w] + dilatedPixels[h-1][w+1]
                + dilatedPixels[h][w-1] + dilatedPixels[h][w] + dilatedPixels[h][w+1]
                + dilatedPixels[h+1][w-1] + dilatedPixels[h+1][w] + dilatedPixels[h+1][w+1]) == val*9:
                    outpt[h][w] = 1
                else:
                    outpt[h][w] = 0
    return outpt


#define Queue class to store vairables during graph/node traversals
class Queue:
   def __init__(self):
       self.items=[]
  
   def isEmpty(self):
       return self.items==[]
      
   def enqueue(self, item):
       self.items.insert(0,item)
      
   def dequeue(self):
       return self.items.pop()
      
   def size(self):
       return len(self.items)
      
q=Queue()

x=[-1,0,1,0]
y=[0,1,0,-1]

# bfs traversal from given (i,j) point
def BFSTraversal(pixel_array, visited, i, j, image_width, image_height, ccimg, count):
   n=0
  
   # add (i,j) into queue adn matk it as visited
   q.enqueue((i,j))
   visited[i][j]=True
  
   # do the following till queue becomes empty
   while(not q.isEmpty()):
       # take a position (a,b) from queue
       a,b=q.dequeue()
       # mark the nvalue at (a,b) in ccimg with component count
       ccimg[a][b]=count
       n+=1
      
       # if any unvisited 1 or 255 values is present in 4 sides of current position, add it into queue
       for z in range(4):
           newI=a+x[z]
           newJ=b+y[z]
           if newI>=0 and newI<image_height and newJ>=0 and newJ<image_width and not visited[newI][newJ] and pixel_array[newI][newJ]!=0:
               visited[newI][newJ]=True
               q.enqueue((newI,newJ))
              
   # at last retun n of values in the current component
   return n


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
   # create visited array and ccimg array where size equal to width and height
   visited=[]
   ccimg=[]
  
   # make all the visited values as False and ccimg values as 0
   for i in range(image_height):
       temp1=[False]*image_width
       temp2=[0]*image_width

       visited.append(temp1)
       ccimg.append(temp2)
  
   ccsizes={}
   count=1
  
   # traverse pixel_array from left to right and from top to bottom
   for i in range(image_height):
       for j in range(image_width):
           # if any unvisited and 1 or 255 value pixel is found, then start bsf traversal from that value
           if not visited[i][j] and pixel_array[i][j]!=0:
               # get number of values in bfs traversal and add it into ccsizes
               n=BFSTraversal(pixel_array, visited, i, j, image_width, image_height, ccimg, count)
               ccsizes[count]=n
               count+=1
              
  
   return (ccimg, ccsizes)

#set all pixels without label assosiated with the largest connected component to 0
def deleteSmallComponents(pixel_array, max_key, image_width, image_height):
    for h in range(image_height):
        for w in range(image_width):
            if pixel_array[h][w] != max_key:
                pixel_array[h][w] = 0
            else:
                pixel_array[h][w] = 255
    return pixel_array

#determine the maximum and minimum x and y values for bounding box
def defBoundingBox(pixel_array, image_width, image_height):
    minX = [image_height, 0]
    maxX = [0, 0]
    minY = [0, image_width]
    maxY = [0, 0]
         
    for h in range(image_height):
        for w in range(image_width):
            if pixel_array[h][w] != 0:
                #define minX
                if (w <= minX[0]):
                    minX = [w,h]
                #define maxX
                if (w >= maxX[0]):
                    maxX = [w,h]
                #define minY
                if (h <= minY[1]):
                    minY = [w,h]
                #define maxY
                if (h >= maxY[1]):
                    maxY = [w,h]
    
    if (minX[1]-5 <= maxX[1] <= minX[1]+5):
        print("SQUARE")
        TL = [minX[0], minY[1]]
        TR = [maxX[0], minY[1]]
        BL = [minX[0], maxY[1]]
        BR = [maxX[0], maxY[1]]

        return (TL,TR,BR,BL)
    else:
        #(TL,TR,BR,BL) 
        return (minX, minY, maxX, maxY)


def main():
    #filename = "./images/covid19QRCode/poster1smallrotated.png"
    #filename = "./images/covid19QRCode/green.png"
    filename = "./images/covid19QRCode/3.png"
    #filename = "./images/covid19QRCode/poster1small.png"
    # DOESNT WORK
    #filename = "./images/covid19QRCode/bch.png"
    #filename = "./images/covid19QRCode/bloomfield.png"
    #misidentifies chair the set
    #filename = "./images/covid19QRCode/playground.png"

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(filename)
    
    #display image
    pyplot.imshow(prepareRGBImageForImshowFromIndividualArrays(px_array_r, px_array_g, px_array_b, image_width, image_height))

    #convert image to greyscale
    pixel_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)

    #compute edge (gradient) magnitude via horizontal and vertical edges
    HorizontalEdgespixel_array = computeHorizontalEdges(pixel_array, image_width, image_height)
    VerticalEdgespixel_array = computeVerticalEdges(pixel_array, image_width, image_height)
    Magnitudepixel_array = computeEdgeMagnitude(HorizontalEdgespixel_array, VerticalEdgespixel_array, image_width, image_height)

    #blur image
    pixel_array = computeGaussianBlur(Magnitudepixel_array, image_width, image_height)
    for i in range(12):
        pixel_array = computeGaussianBlur(pixel_array, image_width, image_height)

    #convert pixles to binary values (either 0 or 255) using thresholding
    pixel_array = computeThreshold(pixel_array, 70, image_width, image_height)

    #fill image component holes with morphological closing
    pixel_array = computeMorphologicalClosing(pixel_array, image_width, image_height)

    #determine the number and location of connected components
    (pixel_array,ccsizes) = computeConnectedComponentLabeling(pixel_array, image_width, image_height)

    #isolate the largest connected component and remove all others
    max_key = max(ccsizes, key=ccsizes.get)
    pixel_array = deleteSmallComponents(pixel_array, max_key, image_width, image_height)

    #define bounding box around isolated component
    (TL,TR,BR,BL)    = defBoundingBox(pixel_array, image_width, image_height)

    # get access to the current pyplot figure
    axes = pyplot.gca()
    

        # create a 70x50 rectangle that starts at location 10,30, with a line width of 3
        #rect = Rectangle( (minX, minY), maxX-minX, maxY-minY, linewidth=3, edgecolor='g', facecolor='none' )
    #    (minX, minY, maxX, maxY)
    print(TL,   TR,   BR,    BL, TL)  


    verts = [TL,TR,BR,BL,TL]
    codes = [ Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY,]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, fill=False, edgecolor='g', lw=2)

    # paint the rectangle over the current plot
    axes.add_patch(patch)

    # plot the current figure
    pyplot.show()



if __name__ == "__main__":
    main()