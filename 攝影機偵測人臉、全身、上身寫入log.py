import cv2  #匯入module
import sys
import time

def write_time_log(body_part, img_counter):   #寫時間log
    file = open('face.txt', 'a')    #要記錄有臉經過的時間
    seconds = time.time()   #抓現在時間
    local_time = time.ctime(seconds)    #轉成時間格式
    now_time = '存圖：' + body_part + '的' + str(img_counter) + '，時間：' + str(local_time) + '\n'
    file.write(now_time)    #寫入檔案
    file.close()   #關檔案

def take_picture(frame, img_counter, body_part): #拍照
    img_name = body_part + 'picture_{}.png'.format(img_counter)
    cv2.imwrite(img_name, frame)
    print('拍照存入' + body_part + str(img_counter).format(img_name))

def camera_on(cap):
    face_img_counter = 0   #圖片標號
    fullbody_img_counter = 0   #圖片標號
    upperbode_img_counter = 0   #圖片標號
    face_pic_time = fullbody_pic_time = upperbody_pic_time = 0 #第一次拍照時間

    while(True):
        ret, frame = cap.read() #從攝影機擷取一張影像，ret代表成功與否，frame是攝影機的單張畫面
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #抓到的影像轉為灰階

        face_detect = faceCascade.detectMultiScale(   #檢測出圖片中所有的人臉，並將人臉用向量保存各個人臉的座標、大小（用矩形表示）
            gray,
            scaleFactor=1.1,    #scaleFactor是指定在圖像大小依每個圖像比例縮小的程度
            minNeighbors=5, #minNeighbors是指定每個候選矩形必須保留多少個鄰居的參數。此參數將影響檢測到的人臉質量，較高的值會導致檢測數量較少但質量較高。
            minSize=(30, 30)    #minSize是最小可能的人臉大小
        )

        if type(face_detect) != tuple:    #如果讀到人臉
            if time.time() - face_pic_time >= 30:  #超過30秒才繼續存log拍照
                face_pic_time = time.time()
                write_time_log('face', face_img_counter)    #寫時間log
                take_picture(frame, face_img_counter, 'face')    #拍照
                face_img_counter += 1
        
        for (x, y, w, h) in face_detect:  #將辨識到的人臉畫框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        fullbody_detect = fullbodyCascade.detectMultiScale(   #檢測出圖片中所有的全身，並將全身用向量保存各個人臉的座標、大小（用矩形表示）
            gray,
            scaleFactor=1.1,    #scaleFactor是指定在圖像大小依每個圖像比例縮小的程度
            minNeighbors=5, #minNeighbors是指定每個候選矩形必須保留多少個鄰居的參數。此參數將影響檢測到的人臉質量，較高的值會導致檢測數量較少但質量較高。
            minSize=(30, 30)    #minSize是最小可能的人臉大小
        )

        if type(fullbody_detect) != tuple:    #如果讀到全身
            if time.time() - fullbody_pic_time >= 30:  #超過30秒才繼續存log拍照
                fullbody_pic_time = time.time()
                write_time_log('fullbody', fullbody_img_counter)    #寫時間log
                take_picture(frame, fullbody_img_counter, 'fullbody')    #拍照
                fullbody_img_counter += 1

        for (x, y, w, h) in fullbody_detect:  #將辨識到的全身畫框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        upperbody_detect = upperbodCascade.detectMultiScale(   #檢測出圖片中所有的上身，並將上身用向量保存各個人臉的座標、大小（用矩形表示）
            gray,
            scaleFactor=1.1,    #scaleFactor是指定在圖像大小依每個圖像比例縮小的程度
            minNeighbors=5, #minNeighbors是指定每個候選矩形必須保留多少個鄰居的參數。此參數將影響檢測到的人臉質量，較高的值會導致檢測數量較少但質量較高。
            minSize=(30, 30)    #minSize是最小可能的上身大小
        )

        if type(upperbody_detect) != tuple:    #如果讀到上身，faces格式會變
            if time.time() - upperbody_pic_time >= 30:  #超過30秒才繼續存log拍照
                upperbody_pic_time = time.time()
                write_time_log('upperbody', upperbode_img_counter)    #寫時間log
                take_picture(frame, upperbode_img_counter, 'upperbode')    #拍照
                upperbode_img_counter += 1

        for (x, y, w, h) in upperbody_detect:  #將辨識到的上身畫框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                
        cv2.imshow('frame', frame)  #顯示圖片

        if cv2.waitKey(1) & 0xFF == ord('q'):   #若按下 q 鍵則離開迴圈
            break

    return 'leave'

while(True):
    casc_face = "haarcascade_profileface.xml"    #臉部偵測檔案
    casc_fullbody = 'haarcascade_fullbody.xml'  #全身偵測檔案
    casc_upperbody = 'haarcascade_upperbody.xml'#上半身偵測檔案
    faceCascade = cv2.CascadeClassifier(casc_face)   #讀取檔案
    fullbodyCascade = cv2.CascadeClassifier(casc_fullbody)   #讀取檔案
    upperbodCascade = cv2.CascadeClassifier(casc_upperbody)   #讀取檔案

    cap = cv2.VideoCapture(0)   #指定抓第0台攝影機
    if not cap.isOpened():  #檢查攝影機是否開著
        print("攝影機沒開，重新偵測")
        time.sleep(10)
        continue
    else:
        cam_run = camera_on(cap)
        if cam_run == 'leave':   #確認camera是否關閉
            cap.release()   #釋放攝影機
            cv2.destroyAllWindows() #關閉所有 OpenCV 視窗
            break