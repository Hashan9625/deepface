from deepface import DeepFace
import cv2
# import matplotlib.pyplot as plt


img1 = cv2.imread('train_img/1/f1.jpg')

img2 = cv2.imread('train_img/4/f3.jpg')


result = DeepFace.verify(img1,img2)

print(result['distance'])
