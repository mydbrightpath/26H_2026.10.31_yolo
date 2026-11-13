from ultralytics import YOLO

# 1. 加载自己训练好的模型。
m_name = "mydataset_sample.pt"
model = YOLO("models/" + m_name)

# 2. 检测一张图片,获得这次预测返回的所有图片结果
results = model.predict(
    source=r"datasets\mydataset_sample\images\test",
    imgsz=640,  # 模型处理图片时使用的输入尺寸
    save=True,  # 希望预测后保存一下结果
    show=False,  # 不用立刻直接显示预测结果
    conf=0.25,  # 过滤掉置信度低于 0.25 的检测结果
    # save_txt = True,#把预测结果保存成txt文件
    # 修改训练结果保存地点：保存在results/yolo11n这个文件夹里面
    project="myresults/predict",
    name=m_name,
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
