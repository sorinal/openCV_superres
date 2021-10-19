import cv2
import matplotlib.pyplot as plt

im = cv2.imread('logo.png')
# print(im.shape)
# plt.imshow(im)
# plt.show()

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel('EDSR_x2.pb')
sr.setModel("edsr", 2)
up = sr.upsample(im)

plt.imshow(up)
plt.show()
