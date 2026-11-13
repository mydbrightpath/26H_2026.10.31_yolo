import cv2 as cv

from ultralytics import YOLO

# 1. 加载自己训练好的模型。
m_name = "mydataset_sample"
model = YOLO("models/" + m_name + ".pt")

# 2. 检测视频、摄像头。获得这次预测返回的所有图片结果
results = model.predict(
    source=0,
    imgsz=640,  # 模型处理图片时使用的输入尺寸
    # 不要等所有画面都处理完再把结果交给我，
    # 而是让我通过循环，逐步取得每一帧的检测结果。
    stream=True,  # 逐步取结果
    save=False,  # 预测后是否保存结果
    show=False,  # 是否立刻直接显示预测结果
    conf=0.25,  # 过滤掉置信度低于 0.25 的检测结果
    # save_txt = True,#把预测结果保存成txt文件
    # 修改训练结果保存地点：保存在results/yolo11n这个文件夹里面
    project="myresults/predict",
    name=m_name,
    # 关闭 Ultralytics 的逐帧推理日志
    verbose=False,  # 自动打印"0: 480x640 (no detections), 7.1ms"这种内容
)

# 3. 遍历检测结果。
for result in results:
    # result.boxes：当前这一个图片中所有检测框的信息。
    # box表示：当前这一个图片中这一个目标的检测信息
    for box in result.boxes:
        # 取出当前目标框的四个坐标：
        # xyxy 表示：左上角 x、左上角 y、右下角 x、右下角 y。
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        # [左上角 x, 左上角 y, 右下角 x, 右下角 y]

        # 取得当前目标的类别编号。
        class_id = int(box.cls.item())  # cls 是 class，也就是“类别”的缩写。
        # 取得当前目标的置信度。
        confidence = float(box.conf.item())
        # 根据类别编号，查到类别名称。
        class_name = result.names[class_id]
        # 假设：result.names 这是 Python 的字典
        # # {0: "apple", 1: "banana", 2: "orange"}

        # 根据检测框计算目标中心位置。
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        print(f"类别：{class_name}")
        print(f"置信度：{confidence:.3f}")
        print(f"检测框：({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        print(f"目标中心：({center_x:.1f}, {center_y:.1f})")
        print("--------------------")

    # ==========================================
    # 用这块代码可以在实时显示摄像头图像或者视频以及显示各信息时
    # 使内存不被占用
    # ==========================================
    # 调用 Ultralytics 的 .plot()，把检测框、类别名称、置信度等画到图片上。
    # 画好的图片存入 plotted。这一行负责画图，还不会弹出窗口。
    plotted = result.plot()

    # 显示画好的图片
    cv.imshow("yolo inference", plotted)

    # cv.waitKey(1)	等待约 1 毫秒，处理窗口事件，并读取按键
    if cv.waitKey(1) & 0xFF == ord("q"):  # ord("q")	获取字符 q 对应的整数编码
        break

# 循环结束后，关闭 OpenCV 创建的窗口。
cv.destroyAllWindows()
