import Detector
import cv2
import SerialPort

DEBUG = True

cap = cv2.VideoCapture(0)
_, image = cap.read()
face_detector = Detector.FaceDetector()
face_judger = Detector.PoseJudge()
driver_state_detector = Detector.DriveDetect()
video_write = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 30.0,
                              (image.shape[1], image.shape[0]))
# serial_port = SerialPort.SerialPort()

# 绘制正方体12轴
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]
person_state = {}

while True:
    t = cv2.getTickCount()
    _, image = cap.read()
    image = cv2.flip(image, 1)

    # 获取脸部特征点
    face_result = face_detector.run(image)
    if face_result:
        pointDict, time = face_result
        reprojectdst, euler_angle = face_judger.getHeadState(pointDict)

        for start, end in line_pairs:
            a = (int(reprojectdst[start][0]), int(reprojectdst[start][1]))
            b = (int(reprojectdst[end][0]), int(reprojectdst[end][1]))
            cv2.line(image, a, b, (0, 0, 255), 2)

        # 最后需要传输的结果
        person_state = driver_state_detector.run(image)
        pose_result = face_judger.judge(pointDict, time)
        # serial_port.send(person_state, pose_result)

        # if person_state is None:
        #     pass
        # else:
        #     pass
        #
        if DEBUG:
            for i in pointDict.keys():
                point = pointDict[i]
                cv2.circle(image, point, 2, (0, 0, 255), -1)
                cv2.putText(image, str(i), point, cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)

            y = 50
            for text in pose_result.keys():
                if pose_result[text]:
                    cv2.putText(image, text, (50, y), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), 1)
                    y += 30

            if person_state:
                for text in person_state.keys():
                    if person_state[text]:
                        cv2.putText(image, text, (50, y), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), 1)
                        y += 30


    print('Time: {}s'.format((cv2.getTickCount() - t) / cv2.getTickFrequency()))
    cv2.imshow('Image', image)
    video_write.write(image)
    if cv2.waitKey(1) == 27:
        break
cap.release()
video_write.release()
