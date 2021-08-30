import matplotlib.pyplot as plt # dung doc 1 buc anh
#matplotlib.pyplot la ham ve do thi
from sklearn.cluster import KMeans
import numpy

img = plt.imread('h.jpg')
height = img.shape[1]
width = img.shape[0] #tra lai 1 tuple - read only -> k thay doi duoc

# (656,561,3) tuple -> k thay doi dc
#[656,651.5]  array/list -> thay doi duoc

print(img.shape)

img = img.reshape(height*width,3)

kmeans = KMeans(n_clusters = 4).fit(img)
labels = kmeans.predict(img) 
clusters = kmeans.cluster_centers_


img2 = numpy.zeros_like(img)
for i in range(len(img2)):
 	img2[i] = clusters[labels[i]] #thay tung diem anh i trong img1 bang cluster tuong tung
img2 = img2.reshape(width,height,3)

plt.imshow(img2)
plt.show()