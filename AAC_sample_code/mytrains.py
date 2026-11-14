"""
大规模训练：
建议先尝试小规模版本，确定能跑通:epochs = 1
然后再改成epochs = 100
选择很多模型，自动连续训练
一个模型训练完之后，自动用下一个模型
"""

from ultralytics import YOLO

ms =  [
    'yolov8n','yolov8s','yolov8m'
    'yolov9t','yolov9s','yolov9m',
    'yolov10n','yolov10s','yolov10m',
    'yolo11n', 'yolo11s', 'yolo11m',
    'yolo12n', 'yolo12s', 'yolo12m',
]


if __name__ == "__main__":
    for m in ms:
        model = YOLO(m + ".pt")
        #开启训练
        model.train(
            data = r"mydataset_sample.yaml",   #打算训练这个数据集
            epochs = 1,#打算训练10轮,把数据集喂给模型10次
            #后面调试会用到的选项
            imgsz = 640,    #全称image size，意思是图片尺寸
            #batch = -1:YOLO会自动帮你找一个相对合适的batch
            batch = -1,  #意思是批一批一批的那个批，batch 指定每个批次的图像数量（batch size）
            cache = "ram",  #意思是缓存
            workers = 1,    #意思是工作者，暗指进程
            #修改训练结果保存地点：分别保存在results/(模型名)这个文件夹里面
            project = "results",
            name = m,
        )