"""
图片或图片文件夹的预测示例：每次运行都保存到新的编号目录。

请把本文件和 result_folders.py 放在同一个文件夹。
在 yolo 环境中运行：python predict_example.py
"""

from pathlib import Path

from ultralytics import YOLO

from result_folders import make_result_folder


# ==================== 只需要优先修改这些参数 ====================

# myresults 的绝对路径。r 表示把 Windows 路径中的反斜杠当作普通字符。
RESULTS_ROOT = Path(r"D:\AAA_deeplearning_ultralytics\ultralytics-8.4.210\myresults")

# 修改为真实模型路径；也可以换成同名的 .pt，目录会自动按格式分开。
MODEL_PATH = Path(r"D:\AAA_deeplearning_ultralytics\ultralytics-8.4.210\models\mydataset_sample.onnx")

# 可以填写单张图片，也可以填写图片文件夹。本示例不处理无限摄像头流。
SOURCE = Path(r"D:\AAA_deeplearning_ultralytics\ultralytics-8.4.210\datasets\mydataset_sample\images\test")

IMAGE_SIZE = 640
CONFIDENCE = 0.25
DEVICE = "cpu"  # 先用 CPU 跑通 ONNX；GPU 需要另行配置兼容的运行环境。

# ==============================================================


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"图片或图片文件夹不存在，请修改 SOURCE：{SOURCE}")

    project, run_name, save_dir = make_result_folder(
        results_root=RESULTS_ROOT,
        mode="predict",
        model_path=MODEL_PATH,
        settings={
            "source": str(SOURCE.resolve()),
            "imgsz": IMAGE_SIZE,
            "conf": CONFIDENCE,
            "device": DEVICE,
        },
    )

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(SOURCE),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        save=True,        # 保存画有检测框的图片。
        show=False,       # 不自动弹窗。
        stream=True,      # 逐步获取结果，避免把整个图片文件夹的结果都存在列表中。
        project=project,  # 上面生成的绝对父目录。
        name=run_name,    # 上面生成的 run_0001、run_0002 等。
        exist_ok=True,   # 允许 YOLO 使用刚刚由工具新建的目录，避免再次自动改名。
    )

    # stream=True 返回生成器，必须遍历，预测和保存才会持续执行。
    try:
        for result in results:
            print(f"{Path(result.path).name}：检测到 {len(result.boxes)} 个目标")
            # 如果需要打印框坐标、类别和置信度，可把之前的逐框循环放在这里。
    finally:
        results.close()

    print(f"预测完成，请打开：{save_dir}")


if __name__ == "__main__":
    main()
