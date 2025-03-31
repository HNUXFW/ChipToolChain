import onnxruntime as ort
import numpy as np
# 加载 VGGONNX 模型
onnx_model_path = "model/VGGsirea5.onnx"
ort_session = ort.InferenceSession(onnx_model_path)
from PIL import Image
image=Image.open("./MNIST/1.jpg")
#将image转化为input大小的格式
image = image.resize((32, 32))
if image.mode != 'RGB':
    image = image.convert('RGB')
image= np.array(image).astype(np.int16)
image = np.transpose(image, (2, 0, 1))
image = np.expand_dims(image, axis=0)
# 进行推理
outputs = ort_session.run(None, {'input': image})
# 输出结果
print("ONNX 模型推理结果：", outputs)