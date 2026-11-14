from ultralytics import YOLO

model = YOLO(r"models\gangqiu1.0_yolo11n.pt")#模型：我想用yolo11n.pt这个模型预测
model.predict(
    source = r"datasets\gangqiu\images\val",#预测目标：我想预测ultralytics\assets里的所有图片
    #预测选项：对预测提一些额外的要求
    save = True,#希望预测后保存一下结果
    show = False,#不用立刻直接显示预测结果
    
    # line_width = 8,#希望预测的时候把线画得粗一点，8个像素点那么粗
    # visualize= True,#可视化特征图(预测过程中产生的一些中间结果),用摄像头的时候可以看到画面以及模型预测的标签
    
    # 选择True是因为现在在用一个"学渣模型"来粗略帮忙打标
    # 这样是为了方便我们直接复制他打出来的YOLO格式的标签
    save_txt = True,#把预测结果保存成txt文件

    #修改训练结果保存地点：保存在results/yolo11n这个文件夹里面
    project = "predict",
    name = "mytest_val_gangqiu1.0_yolo11n",
)