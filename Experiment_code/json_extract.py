import argparse
import json
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

from labelme import utils

# 总结：
# json文件是一个字典，要用到的key：
# imageData（图像数据）
# shapes（是一个列表，列表里面有一个或者多个字典，字典的key：
    # points：多边形点集[[point1],[point2]]
    # label：wood


data = json.load(open('1.json')) # 加载json文件

img = utils.img_b64_to_arr(data['imageData']) # 解析原图片数据
print(data['imageData'])

#lbl = utils.polygons_to_mask(img.shape, data['shapes'][0]['points'])
lbl = utils.shape_to_mask(img.shape, data['shapes'][0]['points'])
draw = np.zeros(img.shape, dtype=np.uint8)


print(data['shapes'][1]['points']) #某一形状
contour = data['shapes'][1]['points']
# contours = []
# contours.append(data['shapes'][1]['points'])
# contours.append(data['shapes'][0]['points'])

print(len(data['shapes']))
#lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes']) #转换成掩膜和标签名
lbl = lbl.astype(np.uint8)
print(img.shape)
cv.imshow('label', lbl*50)
cv.imshow('nihao', img)

# c, _ = cv.findContours(lbl*255, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# print(len(c[0][0][0]))

# print(contours)
for i in range(len(contour)):
    draw = cv.line(draw, tuple(contour[i]), tuple(contour[(i+1)%len(contour)]), color=255, thickness=5)

# draw = cv.drawContours(draw, contours=c, contourIdx=0, color=255, thickness=1)

cv.imshow('draw', draw)

cv.waitKey(0)




