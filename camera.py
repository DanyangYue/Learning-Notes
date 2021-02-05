import cv2
import numpy as np
import os
def show_detections(image,detections):
    h = image.shape[0]
    w = image.shape[1]
    face = []
    lable_map = {0:0, 1:'real',2:'attack'}
    for i in range(detections.shape[2]):
        detections_score = detections[0, 0, i, 2]
        class_id = lable_map[int(detections[0, 0, i, 1])]
        # 与阈值做对比，同一个人脸该过程会进行多次
        if detections_score > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face.append([detections_score, class_id, startX, startY, endX, endY])
            # text = "id:{},{:.2f}%".format(class_id,detections_score * 100)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv2.rectangle(image, (startX, startY), (endX, endY),(255, 255, 255), 2)
            # cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            # cv2.imshow('image',image)
            # cv2.waitKey(250)
            #image,
    #排序，只保留最大置信度的检测框
    n = len(face)
    if n > 0:
        for x in range(n-1):
            for y in range(n-1-x):
                if face[y][0] < face[y+1][0]:
                    face[y],face[y+1] = face[y+1],face[y]
        facescoremax1 = face[0][1:]
        facescoremax = [facescoremax1]
    else:
        facescoremax = face
    return facescoremax

def detect_img(net,image):
    #blob = cv2.dnn.blobFromImage(image, float(1.0), (300, 300), [104.0, 117.0, 123.0], True)
    blob = cv2.dnn.blobFromImage(image, float(0.007843), (300, 300), [127.5], True)
    net.setInput(blob)
    detections = net.forward()
    # print(detections)
    return show_detections(image,detections)

def showimg(image, detect_list):


    for i in detect_list:
        text = i[0]
        startX, startY, endX, endY = i[1], i[2], i[3], i[4]
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),(60, 255, 30), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 80, 255), 2)
    return image

def detector(net):
    
    cap = cv2.VideoCapture(0)
    flag = cap.isOpened()

    index = 1
    while(flag):
        ret, frame = cap.read()
        roi = frame[:,140:500]
        img_gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        # img_gray = cv2.resize(img_gray,(640,480))
        detect_list = detect_img(net,img_gray)
        # print(detect_list)
        img = showimg(roi,detect_list)
        frame[:,140:500] = img
        cv2.rectangle(frame, (140, 0), (500, 480),(0, 0, 255), 2)
        cv2.imshow("Capture_Paizhao",frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):     #按下s键，进入下面的保存图片操作
            cv2.imwrite("./myFacePic/" + str(index) + ".jpg", img_gray)
            print(cap.get(3))
            print(cap.get(4))
            print("save" + str(index) + ".jpg successfuly!")
            print("-------------------------")
            index += 1
        elif k == ord('q'):     #按下q键，程序退出
            break
    cap.release()
    cv2.destroyAllWindows()



def main():

    model_file = "models/ir_face_liveness/mbv1ssd/mobilenet_0109_22w.caffemodel"
    # model_file = "models/ir_face_liveness/mbv1ssd/weight.caffemodel"
    config_file = "models/ir_face_liveness/mbv1ssd/deploy.prototxt"

    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    detector(net)
if __name__ == '__main__':
    main()