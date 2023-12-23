from torchvision.models import *
from visualisation.core.utils import device
import sys
# from efficientnet.model import EfficientNet
sys.path.append("/media/omnisky/a6aeaf75-c964-444d-b0ed-d248f1370cd5/yhq/mlt-Project/csdn_temp/")
from efficientnet.e_utk import EfficientNet


# from efficientnet.e_utk import EfficientNet


# from efficientnet_pytorch import EfficientNet
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch 
from utils import *
import PIL.Image
import cv2

from visualisation.core.utils import device 
from visualisation.core.utils import image_net_postprocessing

from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage
from visualisation.core import *
from visualisation.core.utils import image_net_preprocessing

# for animation

from matplotlib.animation import FuncAnimation
from collections import OrderedDict

# def efficientnet(model_name='efficientnet-b0',**kwargs):
#     return EfficientNet.from_pretrained(model_name).to(device)

max_img = 5
path = '/media/omnisky/a6aeaf75-c964-444d-b0ed-d248f1370cd5/yhq/mlt-Project/csdn_temp/Predict/'


images = [] 

image_paths = glob.glob(f'{path}/*')
category_images = list(map(lambda x: PIL.Image.open(x), image_paths[:max_img]))
images.extend(category_images)


inputs  = [Compose([Resize((224,224)), ToTensor(), image_net_preprocessing])(x).unsqueeze(0) for x in images]  # add 1 dim for batch
inputs = [i.to(device) for i in inputs]



model_outs = OrderedDict()
# model_instances = [
#                   lambda pretrained:efficientnet(model_name='efficientnet-b0')]

# model_names = ['EB0']


images = list(map(lambda x: cv2.resize(np.array(x),(224,224)),images)) # resize i/p img

# for name,model in zip(model_names,model_instances):
name = "mlt"

module = torch.load('/media/omnisky/a6aeaf75-c964-444d-b0ed-d248f1370cd5/yhq/mlt-Project/csdn_temp/UTK_Face/model/efficientnet-b0(utk_mlt_e).pth').cuda()
# module = model(pretrained=True).to(device)
module.eval()

vis = GradCam(module, device)
   

# model_outs[name] = list(map(lambda x: tensor2img(vis(x, None,postprocessing=image_net_postprocessing)[0]), inputs))
# model_outs[name] = list(map(lambda x: tensor2img(vis(x, module._blocks[6]._project_conv,postprocessing=image_net_postprocessing)[0]), inputs))

# model_outs[name] = list(map(lambda x: tensor2img(vis(x, module._blocks[-1]._project_conv,postprocessing=image_net_postprocessing)[0]), inputs))

model_outs[name] = list(map(lambda x: tensor2img(vis(x, module._gender_BLOCKS[-1]._project_conv,postprocessing=image_net_postprocessing)[0]), inputs))

# model_outs[name] = list(map(lambda x: tensor2img(vis(x, module._race_BLOCKS[-1]._project_conv,postprocessing=image_net_postprocessing)[0]), inputs))
    
del module
torch.cuda.empty_cache()

fig, (ax) = plt.subplots(1,1,figsize=(20,20))

    
# def update(frame):
#     all_ax = []
#     ax1.set_yticklabels([])
#     ax1.set_xticklabels([])
#     ax1.text(1, 1, 'Orig. Im', color="white", ha="left", va="top",fontsize=30)

#     plt.imsave('fresh_image.png', images[frame])
   
#     all_ax.append(ax1.imshow(images[frame]))
#     for i,(ax,name) in enumerate(zip(axes,model_outs.keys())):
#         print("---",i)
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])        
#         ax.text(1, 1, name, color="white", ha="left", va="top",fontsize=20)
#         plt.imsave('saved_image.png', model_outs[name][frame])
#         all_ax.append(ax.imshow(model_outs[name][frame], animated=True))

#     return all_ax

# plt.imshow(model_outs[name][frame], animated=True)

# plt.imsave('saved_image.png', img)

for frame in range(len(images)):
    # print(frame)
   
    ax.imshow(model_outs[name][frame])
    ax.axis('off')
    
    plt.savefig(f'frame_{frame}.png', bbox_inches='tight', pad_inches=0, dpi=300)
    ax.clear()

plt.close(fig)


# ani = FuncAnimation(fig, update, frames=range(len(images)), interval=1000, blit=True)
# model_names = [m.__name__ for m in model_instances]
# model_names = ', '.join(model_names)
# fig.tight_layout()
# ani.save('../compare_arch.gif', writer='imagemagick') 