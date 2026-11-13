"""
训练示例：每次新训练保存到新的编号目录。

请把本文件和 result_folders.py 放在同一个文件夹。
在 yolo 环境中运行：python train_example.py
"""

from pathlib import Path

from ultralytics import YOLO

from result_folders import make_result_folder


# ==================== 只需要优先修改这些参数 ====================

RESULTS_ROOT = Path(r"D:\AAA_deeplearning_ultralytics\ultralytics-8.4.210\myresults")

# 这里填写训练开始时加载的 PT 模型：可以是 yolo11n.pt，也可以是自己的 PT。
# 例如用 yolo11n.pt 开始训练，目录中的模型名就是 yolo11n，格式就是 pt。
# 请填写实际存在的文件路径。
MODEL_PATH = Path(r"D:\AAA_deeplearning_ultralytics\ultralytics-8.4.210\models\yolo11n.pt")

# 改为你自己的数据集 YAML 文件路径，文件名也要与实际一致。
DATA_YAML = Path(r"D:\AAA_deeplearning_ultralytics\ultralytics-8.4.210\datasets\mydataset_sample\data.yaml")

EPOCHS = 10       # 训练轮数，请根据你的任务调整。
IMAGE_SIZE = 640 # 输入图片尺寸。
BATCH = 8        # 每批图片数量，显存不足时可减小。
DEVICE = 0       # 第一张 NVIDIA 显卡；也可以设为 "cpu"。

# ==============================================================


def main():
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"数据集配置不存在，请修改 DATA_YAML：{DATA_YAML}")

    project, run_name, save_dir = make_result_folder(
        results_root=RESULTS_ROOT,
        mode="train",
        model_path=MODEL_PATH,
        settings={
            "data": str(DATA_YAML.resolve()),
            "epochs": EPOCHS,
            "imgsz": IMAGE_SIZE,
            "batch": BATCH,
            "device": DEVICE,
        },
    )

    model = YOLO(str(MODEL_PATH))
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=0,       # Windows 新手先使用 0，避免多进程数据加载带来的额外问题。
        save=True,       # 保存训练权重和训练记录。
        project=project, # 使用绝对路径，避免自动加上 runs/detect。
        name=run_name,   # 本次新建目录的编号。
        exist_ok=True,  # 使用工具刚刚新建的目录；工具本身绝不复用旧目录。
    )

    print(f"训练完成，请打开：{save_dir}")
    print(f"训练权重通常位于：{save_dir / 'weights'}")


# Windows 下训练时应保留这个入口保护。
if __name__ == "__main__":
    main()
