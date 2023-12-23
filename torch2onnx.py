import torch
import onnx
from torch.autograd import Variable
# use_gpu = torch.cuda.is_available()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print(torch.__version__)

input_name = ['input']
# output_name = ['output1','output2','output3']
output_name = ['output1']
input = Variable(torch.randn(1, 3, 224, 224)).cuda()

# model = torch.load('mydata/model/efficientnet-b0(order_adience).pth').cuda().eval()
# model.set_swish(memory_efficient=False)
# torch.onnx.export(model, input, 'order_adience.onnx', input_names=input_name, output_names=output_name, verbose=True)
# print("==> Passed")
model = torch.load('mydata/model/efficientnet-b0(b0_data).pth').cuda().eval()
model.set_swish(memory_efficient=False)
torch.onnx.export(model, input, 'b0_data.onnx', input_names=input_name, output_names=output_name, verbose=True)
print("==> Passed")
# test = onnx.load('order_adience.onnx')
# onnx.checker.check_model(test)
# print("==> Passed")
# model_ft = torch.load('mydata/model/efficientnet-b0.pth')
# for i in model_ft.named_parameters():
#     print(i)