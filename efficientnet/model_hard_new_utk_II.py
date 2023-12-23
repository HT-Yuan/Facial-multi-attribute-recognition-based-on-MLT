import re
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)
class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
    def forward(self, x):
        return x.view(x.size()[0],-1)

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params

        self._blocks_args = blocks_args
        #  自己增添了一个gender块
        self._gender = nn.Sequential(OrderedDict([
            ('Conv2d',nn.Conv2d(in_channels = 80,out_channels = 80,
                        kernel_size = (3,3), padding=1)),
            ('BN',nn.BatchNorm2d(num_features=80, momentum=1 - self._global_params.batch_norm_momentum, 
                            eps=self._global_params.batch_norm_epsilon)),
            ('Relu',nn.ReLU()),
            ('AvgPool',nn.AdaptiveAvgPool2d((1, 1))),
            ('view',Reshape()),
            #nn.Dropout(0.5),
            # morph 数据集为2 adience为3
            # utk
            # ('fc',nn.Linear(80,2))
            # adience
            ('fc',nn.Linear(80,2))
        ]))

        # # race 模块
        self._race = nn.Sequential(OrderedDict([
            ('Conv2d',nn.Conv2d(in_channels = 112,out_channels = 112,
                        kernel_size = (3,3), padding=1)),
            ('BN',nn.BatchNorm2d(num_features=112, momentum=1 - self._global_params.batch_norm_momentum, 
                            eps=self._global_params.batch_norm_epsilon)),
            ('Relu',nn.ReLU()),
            ('AvgPool',nn.AdaptiveAvgPool2d((1, 1))),
            ('view',Reshape()),
            #nn.Dropout(0.5),
            # morph 数据集为2 adience为3
            # lfw
            # ('fc',nn.Linear(80,2))
            # adience
            ('fc',nn.Linear(112,5))
        ]))


        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        #对每一层梯度进行归一化
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        # self._gender_BLOCKS = nn.ModuleList([])
        self._race_BLOCKS = nn.ModuleList([])
        
        idx = 0
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            if(idx == 7):
                # self._gender_BLOCKS.append(MBConvBlock(block_args, self._global_params))
                self._race_BLOCKS.append(MBConvBlock(block_args, self._global_params))
            if(idx == 8):
                self._race_BLOCKS.append(MBConvBlock(block_args, self._global_params))
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            
            idx += 1
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                if(idx == 7):
                    # self._gender_BLOCKS.append(MBConvBlock(block_args, self._global_params))
                    self._race_BLOCKS.append(MBConvBlock(block_args, self._global_params))
                if(idx == 8):
                    self._race_BLOCKS.append(MBConvBlock(block_args, self._global_params))
                self._blocks.append(MBConvBlock(block_args, self._global_params))
                idx += 1
        
        

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):  # 公共特征学习 以及 age深层特征
        """ Returns output of the final convolution layer """
        """特征提取网络"""
        

        # Stem This is 直接与输入层相连的第一层卷积 由3*3卷积形成的 3->32(通道数) 由于填充，所以分辨率未变
        # conv(3*3)->bn->swish(激活函数 正则化)
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        # add_yhq
        base_common = inputs
       

        # Blocks
        drop_connect_rate_list = []

        for idx, block in enumerate(self._blocks):
            
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
                drop_connect_rate_list.append(drop_connect_rate)
            x = block(x, drop_connect_rate=drop_connect_rate)

            # 经过测试 blocks个数15个，我认为，对于性别的判断可以提前一点(idx==10 p = 0.936)
            if(idx==6):
                base_common = x
          
                #(16,80,14,14) 
        # extract_gender = self._gender_BLOCKS[0](base_common,drop_connect_rate=drop_connect_rate_list[7])
        extract_gender = base_common
        
        
        
        extract_race = self._race_BLOCKS[0](base_common,drop_connect_rate=drop_connect_rate_list[7])
        
  
        extract_race = self._race_BLOCKS[1](extract_race,drop_connect_rate=drop_connect_rate_list[8])
       
        x = self._swish(self._bn1(self._conv_head(x)))
        

        return x, extract_gender,extract_race
    

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # # Convolution layers 性别识别经过6个blocks后 经自拟的_gender网络
        x_age,extract_gender,extract_race = self.extract_features(inputs)
        x_gender = self._gender(extract_gender)
        x_race = self._race(extract_race)
        
        # Pooling and final linear layer
        x_age = self._avg_pooling(x_age)
        # 平铺为1维
        x_age = x_age.view(bs, -1)

        x_age = self._dropout(x_age)
        x_age = self._fc(x_age)
        # 见证奇迹的时刻
        # lfw
        # x_age = x_age.view(-1, (89-1),2)
        # adience
        # x_age = x_age.view(-1, (8-1),2)
        # pro_a = F.softmax(x_age, dim=2)[:, :, 1]

        # """ 此处只用作 efficientnet_b0 基准测试使用 """
        # x_age,extract_gender = self.extract_features(inputs)
        # # x_gender = self._gender(extract_gender)
        # # Pooling and final linear layer
        # x = self._avg_pooling(x_age)
        # x = x.flatten(start_dim=1)
        # x = self._dropout(x)
        # x = self._fc(x)


        # return x_age,x_gender,pro_a

        # return F.sigmoid(x)
        # return x
        return x_age,x_gender,x_race


    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b' + str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))



###  基于同方差不确定性的任务权重自适应分配
class MTLLoss(nn.Module):
    def __init__(self, task_nums):
        super(MTLLoss, self).__init__()
        x = torch.zeros([task_nums], dtype=torch.float)  # 任务类型
        self.log_var2s = nn.parameter.Parameter(data=x, requires_grad=True)

    def forward(self, L1, L2): # 输入两个损失函
        loss = 0
        for i in range(len(self.log_var2s)):
            # mse = (logit_list[i] - label_list[i]) ** 2
            # print(mse)
            pre = torch.exp(-self.log_var2s[i])
            
            # if i == 0:
            #     loss += torch.sum(pre * L1 + self.log_var2s[i]/2, axis=-1)
            # else:
            #     loss += torch.sum(pre * L2 + self.log_var2s[i]/2, axis=-1)
            

            if i == 0:
                if self.log_var2s[i] < 0 :
                    loss += torch.sum(pre * L1, axis=-1)
                else:
                    loss += torch.sum(pre * L1 + self.log_var2s[i]/2, axis=-1)
            else:
                if self.log_var2s[i] < 0 :
                    loss += torch.sum(pre * L2, axis=-1)
                else:
                    loss += torch.sum(pre * L2 + self.log_var2s[i]/2, axis=-1)
                # loss += torch.sum(pre * L2 + self.log_var2s[i]/2, axis=-1)
            # print(loss)
        return torch.mean(loss)

if __name__ == "__main__":
    # 基于同方差不确定性 的多任务权重自分配
    ml = MTLLoss()
    # x = torch.zeros([2], dtype=torch.float)
    ml(1,2)



    x = torch.zeros([2], dtype=torch.float)
    log_var2s = torch.nn.parameter.Parameter(data=x, requires_grad=True)
    

    print(log_var2s)