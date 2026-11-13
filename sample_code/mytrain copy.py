"""训练备份，以后训练出错了可以用这个备份版本排查问题."""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolo11n.pt")
    # 开启训练
    model.train(
        data=r"coco8.yaml",  # 打算训练coco8这个数据集
        epochs=10,  # 打算训练10轮,把数据集喂给模型10次
        # 后面调试会用到的选项
        imgsz=640,  # 全称image size，意思是图片尺寸
        batch=2,  # 意思是批一批一批的那个批，batch 指定每个批次的图像数量（batch size）
        cache=False,  # 意思是缓存
        workers=0,  # 意思是工作者，暗指进程
        # val = False,    #不验证，单纯就训练；不要这样，验证很重要，有助于训练出更好的模型
    )
