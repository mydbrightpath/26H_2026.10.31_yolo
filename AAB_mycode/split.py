"""
随机划分 YOLO 数据集，将图片和对应的标签一起复制到 train / val / test。

第一次使用，在终端安装进度条工具（不需要 sklearn）：
    python -m pip install tqdm


这里只复制文件，原始图片和标签都保留。一般只需要修改下面的参数。
"""

import math
import random
import shutil
from pathlib import Path

from tqdm import tqdm


# ==================== 可以修改的参数 ====================

# 图片文件夹：例如 abc.jpg 存放在这里。
# 字符串前的 r 用来避免 Windows 路径中的反斜杠被当作转义符。
IMAGES_DIR = Path(r"datasets\gangqiu_origin\images")

# 标签文件夹：例如 abc.jpg 对应的 abc.txt 存放在这里。
LABELS_DIR = Path(r"datasets\gangqiu_origin\labels")

# 划分后的数据集保存位置，脚本会自动创建所需的子文件夹。
OUTPUT_DIR = Path(r"D:\AAL MyCode\AAAmachine vision for diansai\zuoti\26H_2026.10.31_yolo\datasets\gangqiu")

# 三个比例之和必须为 1。下面表示训练集 80%、验证集 10%、测试集 10%。
# 例如想改成 7:2:1，就分别填 0.7、0.2、0.1。
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 固定随机种子：原始文件相同时，同一个种子会得到相同的划分结果。
# 改成其他整数可以换一种随机划分；改成 None 表示每次随机。
RANDOM_SEED = 42

# ======================================================


def split_dataset(images_dir, labels_dir, output_dir,
                  train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """复制并划分数据集，返回各子集的图片数量。"""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    ratios = (train_ratio, val_ratio, test_ratio)
    if any(not math.isfinite(ratio) or not 0 <= ratio <= 1 for ratio in ratios):
        raise ValueError("每个划分比例都必须是 0～1 之间的有限数字。")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("TRAIN_RATIO、VAL_RATIO、TEST_RATIO 三者之和必须为 1。")

    for folder in (images_dir, labels_dir):
        if not folder.is_dir():
            raise FileNotFoundError(f"原始文件夹不存在：{folder}")

    # 输出文件夹必须与原始数据分开，避免把划分结果写进原始数据目录。
    output_path = output_dir.resolve()
    for folder in (images_dir, labels_dir):
        source_path = folder.resolve()
        if (output_path == source_path or source_path in output_path.parents
                or output_path in source_path.parents):
            raise ValueError("输出文件夹和原始文件夹不能相同，也不能互相包含。")

    # 只读取当前目录下的 JPG 图片，不遍历子文件夹；也支持大写 .JPG。
    # 先排序，再随机打乱，保证固定种子可以复现划分。
    images = sorted(
        (p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"),
        key=lambda p: p.name.lower(),
    )
    if not images:
        raise ValueError(f"没有找到 JPG 图片：{images_dir}")

    # 先检查所有图片是否都有同名 TXT 标签，然后才开始复制。
    # 空 TXT 是允许的，YOLO 可以用空标签表示没有目标的图片。
    # 缺少 TXT 时则停止，避免不知情地遗漏数据。
    pairs = []
    missing_labels = []
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if label.is_file():
            pairs.append((image, label))
        else:
            missing_labels.append(label.name)

    if missing_labels:
        examples = "\n".join(missing_labels[:10])
        raise FileNotFoundError(
            f"有 {len(missing_labels)} 张图片缺少同名标签，本次尚未复制任何文件。"
            f"\n缺少的标签（最多显示 10 个）：\n{examples}"
        )

    # 为了防止多次随机划分后混入旧文件，输出目录必须不存在或为空。
    # 不会自动删除、覆盖已有的数据。重新划分时可换一个 OUTPUT_DIR。
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"输出位置不是文件夹：{output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"输出文件夹不为空：{output_dir}\n"
                "请改用一个新的输出文件夹，或先自行移走已有内容再运行。"
            )

    # 每一项包含一张图片及其标签，把整对文件一起打乱，防止对应关系被打乱。
    random.Random(seed).shuffle(pairs)
    total = len(pairs)

    # 按比例分配整数数量：先取整数部分，再把剩余名额给小数部分较大的子集。
    # 这样数量总和始终等于原图片数量，且比例为 0 的子集不会被分到图片。
    expected = [total * ratio for ratio in ratios]
    counts = [int(value) for value in expected]
    remainder = total - sum(counts)
    order = sorted(range(3), key=lambda i: expected[i] - counts[i], reverse=True)
    for index in order[:remainder]:
        counts[index] += 1

    subsets = ("train", "val", "test")
    print(f"共找到 {total} 对图片和标签。")
    for subset, count in zip(subsets, counts):
        print(f"{subset}：{count} 张图片 + {count} 个标签")

    # 自动建立 images/train、labels/train 等六个子文件夹。
    for subset in subsets:
        (output_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)

    start = 0
    with tqdm(total=total, desc="复制数据集", unit="对", dynamic_ncols=True) as progress:
        for subset, count in zip(subsets, counts):
            progress.set_description(f"复制 {subset}")
            for image, label in pairs[start:start + count]:
                # copy2 是复制：保留原文件，并尽可能保留修改时间等信息。
                # 图片和标签使用相同的子集、保留原文件名，标签内容不作修改。
                shutil.copy2(image, output_dir / "images" / subset / image.name)
                shutil.copy2(label, output_dir / "labels" / subset / label.name)
                progress.update(1)  # 复制完一张图片及其标签，进度增加 1。
            start += count

    print(f"\n划分完成，结果保存到：{output_dir}")
    print("原始图片和标签均已保留。没有对应图片的多余标签不会复制。")
    return dict(zip(subsets, counts))


# 直接运行本文件时，从这里开始；导入本文件时不会自动复制数据。
if __name__ == "__main__":
    split_dataset(
        IMAGES_DIR, LABELS_DIR, OUTPUT_DIR,
        TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
    )
