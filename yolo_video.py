import numpy as np
import cv2
import time
import os
import argparse
import imutils

'''
ap = argparse.ArgumentParser()
ap.add_argument("-input","https://youtu.be/nt3D26lrkho",required=True,help="path to input video")
ap.add_argument("-output","F:/pyCharm/OpenCV/video.mp4",required=True,help="path to output video")
args = vars(ap.parse_args())
'''
weightsPath = "D:/2017/Project1/pytorch/yolov3.weights"  # 权重文件
configPath = "D:/2017/Project1/demo/yolov3.txt"  # 配置文件
labelsPath = "D:/2017/Project1/demo/coconames.txt"  # label名称
CONFIDENCE = 0.5  # 过滤弱检测的最小概率
THRESHOLD = 0.4  # 非最大值抑制阈值


#加载标签
with open(labelsPath,'r') as f:
    labels = f.read().rstrip('\n').split('\n')
np.random.seed(42)
COLORS = np.random.randint(0,255,size=(len(labels),3),dtype="uint8")#设置不同颜色

#加载网络，配置权重
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)
print("加载网络，配置权重成功！！！")

#加载视频
vs = cv2.VideoCapture(0)
writer = None
(W,H) = (None,None)

try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] 视频总共有{}帧".format(total))
except:
    print("Error")
    total = -1

#获取每一帧
while True:
    (grabbed,frame) = vs.read()
    if not grabbed:
        break
    if W is None or H is None:
        (H,W) = frame.shape[:2]
    #每一帧送入网络
    blob = cv2.dnn.blobFromImage(frame,1/255.0,(416,416),True,False)
    net.setInput(blob)
    start = time.time()
    outInfo = net.getUnconnectedOutLayersNames()  # # yolo在每个scale都有输出，outInfo是每个scale的名字信息，供net.forward使用
    layerOutputs = net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构。
    end = time.time()
    boxes = []  # 所有边界框（各层结果放一起）
    confidences = []  # 所有置信度
    classIDs = []  # 所有分类ID
    # # 1）过滤掉置信度低的框框
    for out in layerOutputs:  # 各个输出层
        for detection in out:  # 各个框框
            # 拿到置信度
            scores = detection[5:]  # 各个类别的置信度
            classID = np.argmax(scores)  # 最高置信度的id即为分类id
            confidence = scores[classID]  # 拿到置信度

            # 根据置信度筛查
            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes,confidences,CONFIDENCE,THRESHOLD)
    if len(idxs) > 0:
        for i in idxs.flatten():  # indxs是二维的，第0维是输出层，所以这里把它展平成1维
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # 线条粗细为2px
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        writer = cv2.VideoWriter("F:/pyCharm/OpenCV/1.avi",fourcc,30,(frame.shape[1],frame.shape[0]),True)
        if total > 0:
            elap = (end-start)
            print("[INFO] 单帧耗时{:.4f}秒".format(elap))
            print("[INFO] 总耗时约为{:.4f}秒".format(elap*total))
    writer.write(frame)
print("[INFO] cleaning up.....")
writer.release()
vs.release()


