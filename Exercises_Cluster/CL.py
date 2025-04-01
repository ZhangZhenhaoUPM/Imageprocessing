import numpy as np 
import cv2 
# Let's open the image
img_path = 'TestImages/TestImages/Lenna.png'
img= cv2.imread(img_path) 
# We need to re-format the data, we currently have three matrices (3 color values BGR) 
pixel_data = np.float32(img.reshape((-1,3)))
# then perform k-means clustering with random centers
# we can set accuracy to (i.e.) 90 (epsilon)
# and set a maximum number of iterations to 50

number_of_clusters = 2
stop_conds= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.90) 
number_of_attempts = 6
_, regions, centers  = cv2.kmeans(pixel_data, number_of_clusters, None, stop_conds, number_of_attempts , cv2.KMEANS_RANDOM_CENTERS) 
print(regions)
# convert data to image format again again, with its original dimensions
regions = np.uint8(centers)[regions.flatten()]
segmented_image = regions.reshape((img.shape))
# We display original image and result 'segmented' image 
# Probably we need to adjust the number of regions
# And we have to think that we only are considering color information (no neighborhood)
cv2.imshow("original_image", img)
cv2.imshow("segmented_image", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 