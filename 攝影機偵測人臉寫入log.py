import cv2  #匯入module
import sys
import time

def write_time_log(img_counter):   #寫時間log
    file = open('face.txt', 'a')    #要記錄有臉經過的時間
    seconds = time.time()   #抓現在時間
    local_time = time.ctime(seconds)    #轉成時間格式
    now_time = '存圖：' + img_counter + '   時間：' + str(local_time) + '\n'
    file.write(now_time)    #寫入檔案
    file.close()   #關檔案

def take_picture(frame, img_counter): #拍照
    img_name = 'picture_{}.png'.format(img_counter)
    cv2.imwrite(img_name, frame)
    print('拍照存入'.format(img_name))

def camera_on(cap):
    print('im here4')
    img_counter = 0   #圖片標號
    last_pic_time = 0 #第一次拍照時間

    while(True):
        ret, frame = cap.read() #從攝影機擷取一張影像，ret代表成功與否，frame是攝影機的單張畫面
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #抓到的影像轉為灰階
        faces = faceCascade.detectMultiScale(   #檢測出圖片中所有的人臉，並將人臉用向量保存各個人臉的座標、大小（用矩形表示）
            gray,
            scaleFactor=1.1,    #scaleFactor是指定在圖像大小依每個圖像比例縮小的程度
            minNeighbors=5, #minNeighbors是指定每個候選矩形必須保留多少個鄰居的參數。此參數將影響檢測到的人臉質量，較高的值會導致檢測數量較少但質量較高。
            minSize=(30, 30)    #minSize是最小可能的人臉大小
        )

        if type(faces) != tuple:    #如果讀到人臉，faces格式會變
            if time.time() - last_pic_time >= 30:  #超過30秒才繼續存log拍照
                last_pic_time = time.time()
                write_time_log(img_counter)    #寫時間log
                take_picture(frame, img_counter)    #拍照
                img_counter += 1

        for (x, y, w, h) in faces:  #將辨識到的人臉畫框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
        cv2.imshow('frame', frame)  #顯示圖片

        if cv2.waitKey(1) & 0xFF == ord('q'):   #若按下 q 鍵則離開迴圈
            break

    return 'leave'

while(True):
    cascPath = "haarcascade_profileface.xml"    #人臉辨識訓練好的檔案路徑
    faceCascade = cv2.CascadeClassifier(cascPath)   #讀取檔案

    cap = cv2.VideoCapture(0)   #指定抓第0台攝影機
    print('im here')
    if not cap.isOpened():  #檢查攝影機是否開著
        print("攝影機沒開，重新偵測")
        time.sleep(10)
        continue
    else:
        print('im here2')
        cam_run = camera_on(cap)
        print('im here3')
        if cam_run == 'leave':   #確認camera是否關閉
            cap.release()   #釋放攝影機
            cv2.destroyAllWindows() #關閉所有 OpenCV 視窗
            break