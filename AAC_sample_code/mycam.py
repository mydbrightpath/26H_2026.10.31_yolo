from ultralytics import YOLO

import cv2 as cv


model = YOLO(r"D:\AAA_deeplearning_ultralytics\ultralytics-8.3.163\runs\detect\train11\weights\best.pt")#模型：我想用yolo11n.pt这个模型预测
results = model(
    source = 0,#预测目标：电脑摄像头
    
    # 不要等所有画面都处理完再把结果交给我，
    # 而是让我通过循环，逐步取得每一帧的检测结果。
    stream = True,#逐步取结果
)


# 用这块代码可以在实时显示摄像头图像以及显示各信息时
# 使内存不被占用
for result in results:
    # 调用 Ultralytics 的 .plot()，把检测框、类别名称、置信度等画到图片上。
    # 画好的图片存入 plotted。这一行负责画图，还不会弹出窗口。
    plotted = result.plot()

    # 显示画好的图片
    cv.imshow("yolo inference",plotted)
    
    #cv.waitKey(1)	等待约 1 毫秒，处理窗口事件，并读取按键
    if cv.waitKey(1) & 0xff == ord("q"):    #ord("q")	获取字符 q 对应的整数编码
        break

# 循环结束后，关闭 OpenCV 创建的窗口。
cv.destroyAllWindows()