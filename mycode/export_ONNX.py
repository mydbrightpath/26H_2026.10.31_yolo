from ultralytics import YOLO
from pathlib import Path

# 加载模型
model = YOLO("models/mydataset_sample.pt")

# 导出模型,并获取返回的文件路径,导出后原模型保留
# 默认导出到原 .pt 模型所在的文件夹，文件名相同，扩展名变成 .onnx
onnx_path = model.export(format="onnx")

# 打印完整路径，方便找到文件
print("ONNX 保存位置：", Path(onnx_path).resolve())



# # 使用校准数据导出 INT8 量化的 ONNX 模型
# # 创建 'yolo26n_int8.onnx'
# model.export(format="onnx", quantize=8, data="coco8.yaml")  
