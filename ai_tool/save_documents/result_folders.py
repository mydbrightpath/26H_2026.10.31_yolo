"""
训练、预测共用的结果目录管理工具。

本文件只负责创建结果文件夹，不会训练模型，也不会运行预测。
正常使用时，运行 train_example.py 或 predict_example.py 即可。

目录示例：
myresults/predict/mydataset_sample/onnx/run_0001/
"""

import json
from datetime import datetime
from pathlib import Path


def make_result_folder(results_root, mode, model_path, settings=None):
    """新建一个编号目录，返回可传给 YOLO 的 project、name 和完整保存路径。"""
    if mode not in ("train", "predict"):
        raise ValueError('mode 只能是 "train"（训练）或 "predict"（预测）。')

    # resolve() 把路径变成绝对路径，避免 Ultralytics 自动加上 runs/detect。
    results_root = Path(results_root).resolve()
    model_path = Path(model_path).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"找不到模型，请检查 MODEL_PATH：{model_path}")

    # 例如 mydataset_sample.onnx：
    # stem 得到 mydataset_sample；suffix 得到 .onnx。
    model_name = model_path.stem
    model_format = model_path.suffix.lower().lstrip(".")
    if not model_format:
        raise ValueError("模型文件需要有扩展名，例如 .pt 或 .onnx。")
    if mode == "train" and model_format != "pt":
        raise ValueError("这个训练示例需要加载 .pt 模型，不能直接训练 .onnx 模型。")

    # project 是父目录，例如：
    # D:/.../myresults/predict/mydataset_sample/onnx
    project = results_root / mode / model_name / model_format

    # 找出当前父目录中最大的编号，再加 1。
    # 已有 run_0001、run_0002 时，下一次就是 run_0003。
    # 不同模型、不同格式、训练和预测，分别独立编号。
    largest_number = 0
    if project.exists():
        for entry in project.iterdir():
            if entry.name.startswith("run_"):
                number_text = entry.name[4:]
                if number_text.isdigit():
                    largest_number = max(largest_number, int(number_text))

    next_number = largest_number + 1
    while True:
        # :04d 表示至少四位数字，不够的前面补 0，例如 1 -> 0001。
        run_name = f"run_{next_number:04d}"
        save_dir = project / run_name
        try:
            # exist_ok=False：只允许新建，绝不复用已经存在的同名目录。
            # parents=True：自动创建缺少的上级文件夹。
            save_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            # 如果恰好有另一个程序占用了这个编号，换下一个编号继续尝试。
            next_number += 1

    # 在每次结果旁边留一份说明，方便以后确认使用了什么模型和参数。
    # 这是运行信息，不是检测标签文件。
    information = {
        "创建时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "任务": mode,
        "输入模型完整路径": str(model_path),
        "输入模型名称": model_name,
        "输入模型格式": model_format,
        "结果保存路径": str(save_dir),
        "运行参数": settings or {},
    }
    (save_dir / "run_info.json").write_text(
        json.dumps(information, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"本次结果保存到：{save_dir}")
    return str(project), run_name, save_dir
