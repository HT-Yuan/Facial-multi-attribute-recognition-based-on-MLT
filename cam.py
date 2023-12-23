import torch
from torch.autograd import Function
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np

class FeatureExtractor():
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x):
        x.register_hook(self.save_gradient)
        features = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name == self.target_layer:
                x.register_hook(self.save_gradient)
                features += [x]
        return features, x

class ModelOutputs():
    def __init__(self, model, feature_module, target_layer):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layer)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        print(x.size())
        target_activations = []
        # print(self.model._modules.items())
        for name, module in self.model._modules.items():
            print(name)
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)
        return target_activations, x

class GradCam:
    def __init__(self, model, feature_module, target_layer):
        self.model = model
        self.feature_module = feature_module
        self.target_layer = target_layer
        self.model.eval()
        self.extractor = ModelOutputs(self.model, self.feature_module, self.target_layer)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        features, output = self.extractor(input)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_()

        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        cam = target * grads_val
        cam = np.sum(cam, axis=0)

        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        return cam

# 加载预训练的 EfficientNet 模型

model = torch.load('UTK_Face/model/efficientnet-b0(utk_mlt_e).pth') 

# 选择要可视化的目标层
target_layer = "blocks[-1]._expand_conv"
# feature_module = eval("model." + target_layer)

# 初始化 Grad-CAM
grad_cam = GradCam(model=model, feature_module=model._blocks[-1]._expand_conv, target_layer=target_layer)

# 读取并预处理输入图像
image_path = "/media/omnisky/a6aeaf75-c964-444d-b0ed-d248f1370cd5/yhq/mlt-Project/csdn_temp/Predict/39_0_0_20170117165645001.jpg"
img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.transpose(img, (2, 0, 1))
img = torch.FloatTensor(img).unsqueeze(0).cuda()

# 获取 Grad-CAM 热图
cam = grad_cam(img)

# 将热图叠加到原始图像上
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
img = np.transpose(img.squeeze(), (1, 2, 0))
cam = heatmap + np.float32(img)
cam = cam / np.max(cam)

# 显示或保存 Grad-CAM 可视化结果
cv2.imwrite("gradcam.jpg", np.uint8(255 * cam))