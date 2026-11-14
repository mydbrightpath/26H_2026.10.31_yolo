from ultralytics import YOLO
import cv2 as cv
import time

#在linux终端上输入这行命令:echo $DISPLAY 输出的所有结果赋值过来,例如:'localhost:11.0'
#这样就可以把图像直接显示出来
# os.environ['DISPLAY'] = 'localhost:10.0'


# 打开 USB 摄像头
# 参数 0 通常代表电脑默认摄像头或插入的第一个 USB 摄像头
# 如果打不开，可以尝试改成 1 或 2
# cap = cv.VideoCapture('/dev/video0',cv.CAP_V4L2)  #linux系统上的打开方法
cap = cv.VideoCapture(1)    #电脑上的打开方法


# 检查摄像头是否成功打开
if not cap.isOpened():
    print(' Error: 无法打开 USB 摄像头！' \
    '请检查 USB 连接或修改设备索引号 (如改用 cap = cv.VideoCapture(1))')
    exit()

# 可选：设置摄像头的分辨率（例如设置为 640x480保证处理流畅度，还可设置为1080p:1920x1080）
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

# 1. 加载自己训练好的模型。
m_name = "gangqiu1.0_yolo11n"
model = YOLO("models/" + m_name + ".pt")

try:

    flag = 0
    exitflag = 0
    frame_count = 0
    fps = 0.0

    while True:
        # 逐帧读取图像
        # ret: bool 类型，True 表示成功读取到一帧图片；frame: 当前帧的彩色图像矩阵 (BGR)
        ret, img = cap.read()

        # 用一段时间内处理的帧数计算平均 FPS，避免数字跳动过大。
        start_time = time.perf_counter()

        # 2. 获得这次预测返回的所有图片结果
        results = model.predict(
            source=img, #来源是摄像头读取到的每一帧图片
            imgsz=640,      # 模型处理图片时使用的输入尺寸

            # 不要等所有画面都处理完再把结果交给我，
            # 而是让我通过循环，逐步取得每一帧的检测结果。
            stream = True,#逐步取结果

            save = False,   #预测后是否保存结果
            show = False,    #是否立刻直接显示预测结果
            conf=0.25,     # 过滤掉置信度低于 0.25 的检测结果
            # save_txt = True,#把预测结果保存成txt文件

            #修改训练结果保存地点：保存在results/yolo11n这个文件夹里面
            project = "predict",
            name = m_name+"_cam1",

            # 关闭 Ultralytics 的逐帧推理日志
            verbose=False,  #自动打印"0: 480x640 (no detections), 7.1ms"这种内容
        )

        # 为什么放在这里读出的帧数不对
        # frame_count += 1
        # # 计算距离上次更新 FPS 过了多少秒。
        # now = time.perf_counter()
        # elapsed = now - start_time
        # # 每隔约 1 秒更新一次 FPS。
        # if elapsed >= 1.0:
        #     fps = frame_count / elapsed
        #     frame_count = 0
        #     start_time = now
        
        


        # 3. 遍历检测结果。
        for result in results:
            flag += 1

            # result.boxes：当前这一个图片中所有检测框的信息。
            # box表示：当前这一个图片中这一个目标的检测信息
            for box in result.boxes:    
                # 取出当前目标框的四个坐标：
                # xyxy 表示：左上角 x、左上角 y、右下角 x、右下角 y。
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # [左上角 x, 左上角 y, 右下角 x, 右下角 y]

                # 取得当前目标的类别编号。
                class_id = int(box.cls.item())  #cls 是 class，也就是“类别”的缩写。
                # 取得当前目标的置信度。
                confidence = float(box.conf.item())
                # 根据类别编号，查到类别名称。
                class_name = result.names[class_id]
                # 假设：result.names 这是 Python 的字典
                # # {0: "apple", 1: "banana", 2: "orange"}

                # 根据检测框计算目标中心位置。
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2


                #计算帧率FPS
                frame_count += 1
                # 计算距离上次更新 FPS 过了多少秒。
                now = time.perf_counter()
                elapsed = now - start_time
                # 每隔约 1 秒更新一次 FPS。
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = now

                # 使用flag延缓打印速度，使得能够看清楚输出
                if flag == 10:
                    print(f"类别：{class_name}")
                    print(f"置信度：{confidence:.3f}")
                    print(f"检测框：({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                    print(f"目标中心：({center_x:.1f}, {center_y:.1f})")
                    print(1/elapsed)
                    print(fps)  #这个不对为什么
                    print("--------------------")
                    flag = 0

            # ==========================================
            # 用这块代码可以在实时显示摄像头图像或者视频以及显示各信息时
            # 使内存不被占用
            # ==========================================
            # 调用 Ultralytics 的 .plot()，把检测框、类别名称、置信度等画到图片上。
            # 画好的图片存入 plotted。这一行负责画图，还不会弹出窗口。
            plotted = result.plot()

            # 在图片左上角显示 FPS。
            cv.putText(
                plotted,
                f"FPS: {1/elapsed}",
                (20, 40),               # 文字左下角的位置
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,                    # 字体大小
                (0, 255, 0),            # BGR 颜色：绿色
                2,                      # 文字粗细
            )

            # 显示画好的图片
            cv.imshow("yolo inference",plotted)

            #cv.waitKey(1)	等待约 1 毫秒，处理窗口事件，并读取按键
            if cv.waitKey(1) & 0xFF == ord("q"):    #ord("q")	获取字符 q 对应的整数编码
                exitflag = 1
                break

        #上面的键盘输入q退出的是内层循环,退出不了外层的，所以无法实现关闭
        # 这个部分是为了实现关闭而添加的
        if exitflag == 1:
            exitflag = 0
            break

finally:
     # 正常退出或出现异常时，都释放摄像头并关闭窗口。
    cap.release()
    cv.destroyAllWindows()



