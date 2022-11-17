import random
from math import exp, pow
import cv2.cv2
import cv2.dnn
import numpy as np

className = list(map(lambda x: x.strip(), open('coco.names', 'r').readlines())) # 可替换自己的类别文件
# 替换对应yolo的Anchors值
netAnchors = np.asarray([[12.0, 16.0, 19.0, 36.0, 40.0, 28.0],
                         [36.0, 75.0, 76.0, 55.0, 72.0, 146.0],
                         [142.0, 110.0, 192.0, 243.0, 459.0, 401.0]], dtype=np.float32)

netStride = np.asarray([8.0, 16.0, 32.0], dtype=np.float32)
netWidth = 640
netHeight = 640
nmsThreshold = 0.80
boxThreshold = 0.80
classThreshold = 0.80


def GetColors(color_num):
    ColorList = []
    for num in range(color_num):
        R = random.randint(100, 255)
        G = random.randint(100, 255)
        B = random.randint(100, 255)
        BGR = (B, G, R)
        ColorList.append(BGR)
    return ColorList


def Sigmoid(x):
    x = float(x)
    out = (float(1.) / (float(1.) + exp(-x)))
    return float(out)


def readModel(netPath, isCuda=False):
    try:
        net = cv2.dnn.readNetFromONNX(netPath)
    except:
        return False

    if isCuda:  # GPU
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    else:  # CPU
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def Detect(SrcImg, net, netWidth, netHeight):
    netInputImg = SrcImg
    blob = cv2.dnn.blobFromImage(netInputImg, scalefactor=1 / 255.0, size=(netWidth, netHeight), mean=[104, 117, 123], swapRB=True, crop=False)
    net.setInput(blob)  # 把图像放入网络
    netOutputImg = net.forward(net.getUnconnectedOutLayersNames())  # 获取输出结果集

    netOutputImg = np.array(netOutputImg, dtype=np.float32)
    ratio_h = float(netInputImg.shape[1] / netHeight)
    ratio_w = float(netInputImg.shape[0] / netWidth)
    pdata = netOutputImg[0]
    classIds = []  # 定义结果线性表
    confidences = []  # 定义置信度线性表
    boxes = []  # 定义坐标线性表
    count = 0
    for stride in range(3):  # netStride = {8.0, 16.0, 32.0} = 3
        grid_x = netWidth / netStride[stride]
        grid_y = netHeight / netStride[stride]
        grid_x, grid_y = int(grid_x), int(grid_y)  # 系统默认是float32，这里是为了下面的循环转为int

        for anchor in range(3):  # netAnchors 的层数 = 3
            anchor_w = netAnchors[stride][anchor * 2]
            anchor_h = netAnchors[stride][anchor * 2 + 1]
            anchor_w, anchor_h = float(anchor_w), float(anchor_h)

            for i in range(grid_x):
                for j in range(grid_y):  # 到这的下一行总运行次数是25200 = (80*80*3) + (40*40*3) + (20*20*3)
                    # 到这了是需要运行25200次
                    pdatabox = pdata[0][count][4]
                    box_score = Sigmoid(pdatabox) # 获取每一行的box框中含有某个物体的概率
                    if box_score > boxThreshold:  # box的阈值起作用了
                        scores = pdata[0][count][5:]  # 这里的scores理应是一个多维矩阵
                        _, max_class_socre, _, classIdPoint = cv2.minMaxLoc(scores)  # 求最大值以及最大值的位置&位置是元组
                        max_class_socre = np.asarray(max_class_socre, dtype=np.float64)
                        max_class_socre = Sigmoid(max_class_socre)
                        if max_class_socre > classThreshold:  # 类别的置信度起作用
                            # rect[x,y,w,h]
                            pdatax=pdata[0][count][0]
                            x = (Sigmoid(pdatax) * float(2.) - float(0.5) + j) * netStride[stride]  # x
                            pdatay = np.asarray(pdata[0][count][1], dtype=np.float64)
                            y = (Sigmoid(pdatay) * float(2.) - float(0.5) + i) * netStride[stride]  # y
                            pdataw=pdata[0][count][2]
                            w = pow(Sigmoid(pdataw) * float(2.), float(2.0)) * anchor_w  # w
                            pdatah = pdata[0][count][3]
                            h = pow(Sigmoid(pdatah) * float(2.), float(2.0)) * anchor_h  # h
                            left = (x - 0.5 * w) * ratio_w
                            top = (y - 0.5 * h) * ratio_h
                            left, top, W, H = int(left), int(top), int(w * ratio_w), int(h * ratio_h)
                            # 对classIds & confidences & boxes
                            classIds.append(classIdPoint[1])  # 获取最大值的位置
                            confidences.append(max_class_socre * box_score)
                            boxes.append((left, top, W, H))
                    count += 1
    # cv2.dnn.NMSBoxes的bboxes框应该是左上角坐标(x,y)和 w，h， 那么针对不同模型的，要具体情况转化一下，才能应用该函数。
    nms_result = cv2.dnn.NMSBoxes(boxes, confidences, classThreshold, nmsThreshold)  # 抑制处理 返回的是一个数组
    result_id, result_confidence, result_boxs = [], [], []
    for idx in nms_result:
        result_id.append(classIds[idx])
        result_confidence.append(confidences[idx])
        result_boxs.append(boxes[idx])
    if len(result_id) == 0:
        return False
    else:
        return result_id, result_confidence, result_boxs


def drawPred(img, result_id, result_confidence, result_boxs, color):
    for i in range(len(result_id)):
        class_id = result_id[i]
        confidence = round(result_confidence[i], 2)  # 保留两位有效数字
        box = result_boxs[i]
        pt1 = (box[0], box[1])
        pt2 = (box[0] + box[2], box[1] + box[3])
        # 绘制图像目标位置
        cv2.rectangle(img, pt1, pt2, color[i], 2, 2)  # x1, y1, x2, y2 = box[0], box[1], box[0] + box[2], box[1] + box[3]
        cv2.rectangle(img, (box[0], box[1]-18), (box[0] + box[2], box[1]), color[i], -1)

        label = "%s:%s" % (className[class_id], confidence)  # 给目标进行添加类别名称以及置信度
        FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, label, (box[0] - 2, box[1] - 5), FONT_FACE, 0.5, (0, 0, 0), 1)
    cv2.imwrite("my640.jpg", img)
    cv2.imshow("outPut", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    img_path = "C:/Users/kiven/Desktop/B-yolov7-c++/640.jpg"
    model_path = "C:/Users/kiven/Desktop/B-yolov7-c++/mymodels1.onnx"
    classNum = 80
    color = GetColors(classNum)
    Mynet = readModel(model_path, isCuda=False)
    img = cv2.imread(img_path)
    result_id, result_confidence, result_boxs = Detect(img, Mynet, 640, 640)
    drawPred(img, result_id, result_confidence, result_boxs, color)
