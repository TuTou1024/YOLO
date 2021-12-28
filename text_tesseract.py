import pytesseract
from pytesseract import Output
import cv2
import argparse

#参数初始化
ap = argparse.ArgumentParser()
ap.add_argument("--image", type=str, default="D:/2017/Project1/horse.jpg", help="path to the image")
ap.add_argument("--minconfig", type=int, default=0, help="mininum confidence value to filter weak text detection")
args = vars(ap.parse_args())

#加载图片并确定图片文本位置
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_data(rgb,output_type=Output.DICT)

#遍历每一个单独的本地化文本
for i in range(0,len(results["text"])):
    #提取文本区域的边界框坐标
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]

    #提取文本本身以及本地化文本的置信度
    text = results["text"][i]
    conf = int(results["conf"][i])
    if conf > args["minconfig"]:
        print("置信度为{}".format(conf))
        print("文本为{}".format(text))
        print("")
        #putText函数不支持非ASCII码，所以要删掉文本中非ASCII码
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
cv2.imshow("Image", image)
cv2.waitKey(0)