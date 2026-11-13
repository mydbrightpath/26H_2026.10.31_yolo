"""只用一种模型"""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"models\yolo11n.pt")
    #开启训练
    model.train(
        data = r"mydataset_sample.yaml",   #打算训练coco8这个数据集
        epochs = 10,#打算训练10轮,把数据集喂给模型10次
        #后面调试会用到的选项
        imgsz = 640,    #全称image size，意思是图片尺寸
        #batch = 2:一次喂两张图片
        #batch = -1:YOLO会自动帮你找一个相对合适的batch
        batch = -1,  #意思是批一批一批的那个批，batch 指定每个批次的图像数量（batch size）
        cache = "ram",  #意思是缓存
        workers = 1,    #意思是工作者，暗指进程
        # #希望用CPU来训练（不行）
        # device = "cpu",
        #修改训练结果保存地点：保存在results/yolo11n这个文件夹里面
        project = "myresults/train",
        name = "yolo11n",
    )
    