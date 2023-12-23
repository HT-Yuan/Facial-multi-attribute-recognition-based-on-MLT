from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.nn import functional as F
import time
from sklearn.metrics import precision_recall_fscore_support as score
from efficientnet.model import EfficientNet
import os

# some parameters

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_gpu = torch.cuda.is_available()

data_dir = 'mydata'
answer_dir = 'mydata/answer'
batch_size = 32
# lr = 0.01  #
lr = 0.01
momentum = 0.9
num_epochs = 80
input_size = 224

net_name = 'efficientnet-b0'

# efficientnet迁移学习后得预训练权重
pth_map = {
    'efficientnet-b0': 'efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'efficientnet-b7-dcc49843.pth',
}

# 数据集的预处理，以及数据载入（分批次）
def loaddata(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        'train': transforms.Compose([
            #转为灰度
            transforms.Grayscale(num_output_channels=3),
            #保留了纵横比
            transforms.Resize(input_size),
            #正方形 消除纵横比
            transforms.CenterCrop(input_size),
            # 数据增强
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # datasets.ImageFolder：输入文件夹/data/train 预处理方式 
    # image_datasets为一个字典 key:test/train value
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
    #class_to_idx

    
    # num_workers=0 if CPU else =1
    # 将图像和标签分别封装成一个Tensor 并且在这里分了batch_size
    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=shuffle, num_workers=1) for x in [set_name]}
    
    data_set_sizes = len(image_datasets[set_name])
    
    return dataset_loaders, data_set_sizes

# 多属性数据集(UTK Face)的预处理，以及数据载入（分批次）
def loaddata_utk(data_dir, batch_size, set_name, shuffle):
    data_transforms = {
        'train': transforms.Compose([
            #转为灰度
            transforms.Grayscale(num_output_channels=3),
            #保留了纵横比
            transforms.Resize(input_size),
            #正方形 消除纵横比
            transforms.CenterCrop(input_size),
            # 数据增强
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    # datasets.ImageFolder：输入文件夹/data/train 预处理方式 
    # image_datasets为一个字典 key:test/train value
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
    #class_to_idx

    
    # num_workers=0 if CPU else =1
    # 将图像和标签分别封装成一个Tensor 并且在这里分了batch_size
    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                      batch_size=batch_size,
                                                      shuffle=shuffle, num_workers=1) for x in [set_name]}
    
    data_set_sizes = len(image_datasets[set_name])
    
    return dataset_loaders, data_set_sizes

def cost_fn(logits, levels, imp=1):
    temp = (F.log_softmax(logits, dim=2)[:, :, 1]*levels
                      + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels))*imp
    # val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1]*levels
    #                   + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels))*imp, dim=1))
    # print(1-levels)
    # print(levels)
    # print(F.log_softmax(logits, dim=2))
    # print(F.log_softmax(logits, dim=2)[:, :, 1]*levels)
    # print(F.log_softmax(logits, dim=2)[:, :, 1])
    # print(temp)
    # print(levels)
    val = (-torch.sum(temp, dim=1))
    # print(val) # 理论来说 这里都应该是正数
    return torch.mean(val)


def train_model(model_ft, criterion, optimizer, lr_scheduler, num_epochs=50, a = 0.1):
    # answer.txt用作存储训练结果
    file = open(os.path.join(answer_dir,'answer('+str(a)+').txt'),'w')
    since = time.time()
    # 最好的训练权重，初始值为预训练权重
    best_model_wts = model_ft.state_dict()
    # 最好的准确度
    best_acc = 0.0
    # 启用 BatchNormalization 和 Dropout 这是因为用作训练
    model_ft.train(True)

    for epoch in range(num_epochs):
        dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=batch_size, set_name='train', shuffle=True)
        print('Data Size', dset_sizes)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        
        print('-' * 10)
        #优化器与学习律的结合
        optimizer = lr_scheduler(optimizer, epoch)
        # 计算每一次迭代的损失值
        running_loss_age = 0.0
        running_loss_gender = 0.0
        # 预测成功的个数 用作计算准确度
        running_corrects_age=0
        running_corrects_gender = 0
        # 计算批次数
        count = 0


        for data in dset_loaders['train']:
            inputs, labels = data
            labels = torch.squeeze(labels.type(torch.LongTensor))
            # 由总标签到各独立任务标签
            label_age = (labels+1)/2
            # 将labele_age转为子二分类标签（为损失函数做准备）
            # print(label_age.size())
            levels = []
            for i in range(label_age.size()[0]):
                # print(label_age[i]) 
                #levels.append([1]*(label_age[i]) + [0]*(8 - 1 - label_age[i]))  torch版本问题
                levels.append([1]*int(label_age[i]) + [0]*int(8 - 1 - label_age[i])) 
            levels = torch.tensor(levels, dtype=torch.uint8)

            # u->0 female->1 male->1
            label_gender_index = torch.ne(labels,0)
            label_gender = (labels%2+1)*(label_gender_index)
            # 标签转换完毕
            # 将数据转换为gpu训练的格式 
            inputs, label_age,label_gender,l_agehot = Variable(inputs.cuda()), Variable(label_age.cuda()),Variable(label_gender.cuda()),Variable(levels.cuda())
            # 消融实验
            # inputs, label_age,label_gender = Variable(inputs.cuda()), Variable(label_age.cuda()),Variable(label_gender.cuda())
            # 模型加载 读取输出值
            output_age,output_gender,pro_age = model_ft(inputs)
            # 消融
            # output_age,output_gender = model_ft(inputs)
            # 损失值
            loss_age = cost_fn(output_age, l_agehot)*a
            # print("IIII: ",loss_age,output_age, l_agehot)
            # 消融实验
            # loss_age = criterion(output_age, label_age)*0.5
            loss_gender = criterion(output_gender, label_gender)*(1-a)
            # 预测值
            pro_age = pro_age > 0.5
            preds_age = torch.sum(pro_age, dim=1)
            # 消融
            # _, preds_age = torch.max(output_age.data, 1)

            _, preds_gender = torch.max(output_gender.data, 1)

            #反向传播，更新梯度
            optimizer.zero_grad()
            loss_age.backward(retain_graph=True)
            loss_gender.backward()
            optimizer.step()

            count += 1
            if count % 30 == 0 or output_gender.size()[0] < batch_size:
                print('Epoch:{}: loss_age:{:.3f}, loss_gender:{:.3f}'.format(epoch, loss_age.item(),loss_gender.item()))

            running_loss_age += loss_age.item()*inputs.size(0)
            running_loss_gender += loss_gender.item()*inputs.size(0)
            running_corrects_age += torch.sum(preds_age == label_age.data)
            running_corrects_gender += torch.sum(preds_gender == label_gender.data)


        # 每次迭代损失值、准确度相关数据计算
        epoch_acc_age = running_corrects_age.double() / dset_sizes
        epoch_acc_gender = running_corrects_gender.double() / dset_sizes
        epoch_loss_age = running_loss_age/dset_sizes
        epoch_loss_gender  = running_loss_gender/dset_sizes
        print('loss_age: {:.4f} loss_gender: {:.4f} Acc_age: {:.4f} Acc_gender: {:.4f}'.format(
            epoch_loss_age, epoch_loss_gender,epoch_acc_age, epoch_acc_gender))

        file.write(str(epoch) + '\t' + str(float(epoch_loss_age)) + '\t'+str(float(epoch_loss_gender)) + '\t'+
                    str(float(epoch_acc_age))+'\t'+str(float(epoch_acc_gender))+'\n')

        if epoch_acc_age > best_acc:
            best_acc = epoch_acc_age
            best_model_wts = model_ft.state_dict()
        if epoch_acc_age > 0.999 and epoch_acc_gender>0.999:
            break

    # save best model'
    file.close()
    save_dir = data_dir + '/model'
    model_ft.load_state_dict(best_model_wts)
    model_out_path = save_dir + "/" + net_name + '('+str(a)+').pth'
    torch.save(model_ft, model_out_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return best_model_wts



def test_model(model, criterion,a=0.1):
    file = open(os.path.join(answer_dir,'answer('+str(a)+').txt'),'a')

    model.eval()
    # 损失值
    running_loss_age = 0.0
    running_loss_gender = 0.0
    #预测正确的个数
    running_corrects_age = 0
    running_corrects_gender = 0
    # 存储总的预测值
    outPre_age = []
    outLabel_age = []
    outPre_gender = []
    outLabel_gender = []
    mseloss_age = 0
    # 读取每批次数据
    dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=16, set_name='test', shuffle=False)
    for data in dset_loaders['test']:
        inputs, labels = data
        labels = torch.squeeze(labels.type(torch.LongTensor))
        # 由总标签到各独立任务标签
        label_age = (labels+1)/2
        # 将labele_age转为子二分类标签（为损失函数做准备）
        levels = []
        for i in range(label_age.size()[0]):
            # levels.append([1]*label_age[i] + [0]*(8 - 1 - label_age[i]))
            levels.append([1]*int(label_age[i]) + [0]*int(8 - 1 - label_age[i])) 
        levels = torch.tensor(levels, dtype=torch.uint8)
        # u->0 female->1 male->1
        label_gender_index = torch.ne(labels,0)
        label_gender = (labels%2+1)*(label_gender_index)
        # 将数据转换为gpu训练的格式 
        inputs, label_age,label_gender,l_agehot = Variable(inputs.cuda()), Variable(label_age.cuda()),Variable(label_gender.cuda()),Variable(levels.cuda())
        # 消融实验
        # inputs, label_age,label_gender = Variable(inputs.cuda()), Variable(label_age.cuda()),Variable(label_gender.cuda())
        #读取的输出值
        output_age, output_gender,pro_age = model_ft(inputs)
        # 消融实验
        # output_age, output_gender = model_ft(inputs)
        #计算损失
        loss_age = cost_fn(output_age, l_agehot)*a
        #消溶
        # loss_age = criterion(output_age, label_age)*0.5
        loss_gender = criterion(output_gender, label_gender)*(1-a)
        # 将输出值转为预测值
        # pro_age = pro_age > 0.5
        # preds_age = torch.sum(pro_age, dim=1)
        _, preds_gender = torch.max(output_gender.data, 1)
        #消溶
        _, preds_age = torch.max(output_age.data, 1)

        # 将每批次预测值链接起来，方便最终计算其他指标
        outPre_age.extend(preds_age.data.cpu().tolist())
        outLabel_age.extend(label_age.data.cpu().tolist())
        outPre_gender.extend(preds_gender.data.cpu().tolist())
        outLabel_gender.extend(label_gender.data.cpu().tolist())

        # 损失值
        running_loss_age += loss_age.item()*inputs.size(0)
        running_loss_gender += loss_gender.item()*inputs.size(0)
        # 准确度
        running_corrects_age += torch.sum(preds_age == label_age.data)
        running_corrects_gender += torch.sum(preds_gender == label_gender.data)
        mseloss_age += sum((preds_age - label_age).pow(2).tolist())
    #这里在终端输出每次模型的评价指标
    file.write('Loss_age: {:.4f}  Loss_gender: {:.4f} Acc_age: {:.4f} Acc_gender: {:.4f} mseloss_age: {:.4f} '.format(running_loss_age/dset_sizes,running_loss_gender/dset_sizes,
                                            running_corrects_age.double() / dset_sizes,running_corrects_gender.double() / dset_sizes, mseloss_age /dset_sizes))
    print('Loss_age: {:.4f}  Loss_gender: {:.4f} Acc_age: {:.4f} Acc_gender: {:.4f} mseloss_age: {:.4f} '.format(running_loss_age/dset_sizes,running_loss_gender/dset_sizes,
                                            running_corrects_age.double() / dset_sizes,running_corrects_gender.double() / dset_sizes, mseloss_age /dset_sizes))
    # p\r\f\总数
    precision_a, recall_a, fscore_a, support_a = score(outLabel_age, outPre_age)
    precision_g, recall_g, fscore_g, support_g = score(outLabel_gender, outPre_gender)
    print('precision_a: {}'.format(precision_a))
    print('recall_a: {}'.format(recall_a))
    print('fscore_a: {}'.format(fscore_a))
    print('support_a: {}'.format(support_a))
    print('precision_g: {}'.format(precision_g))
    print('recall_g: {}'.format(recall_g))
    print('fscore_g: {}'.format(fscore_g))
    print('support_g: {}'.format(support_g))

    file.write('precision_a: {}'.format(precision_a))
    file.write('recall_a: {}'.format(recall_a))
    file.write('fscore_a: {}'.format(fscore_a))
    file.write('support_a: {}'.format(support_a))
    file.write('precision_g: {}'.format(precision_g))
    file.write('recall_g: {}'.format(recall_g))
    file.write('fscore_g: {}'.format(fscore_g))
    file.write('support_g: {}'.format(support_g))

    file.close()
    


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0025, lr_decay_epoch=10):
    """Decay learning rate by a f#            model_out_path ="./model/W_epoch_{}.pth".format(epoch)
#            torch.save(model_W, model_out_path) actor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.8**(epoch // lr_decay_epoch))

    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

# 对预训练权重就行修改（网络有修改）
def add_dict(state_dict):
    state_dict['_gender.Conv2d.weight'] = torch.randn(([80, 80, 3, 3]))
    state_dict['_gender.Conv2d.bias'] = torch.randn(([80]))
    state_dict['_gender.BN.weight'] = torch.randn(([80]))
    state_dict['_gender.BN.bias'] = torch.randn(([80]))
    state_dict['_gender.BN.running_mean'] = torch.randn(([80]))
    state_dict['_gender.BN.running_var'] = torch.randn(([80])) 
    state_dict['_gender.fc.weight'] = torch.randn(([3,80]))
    state_dict['_gender.fc.bias'] = torch.randn(([3]))
    return state_dict

if __name__ == "__main__":


    # # # 加载网络框架
    # a_temp = 0.1
    # while(a_temp<=1.0):
    #     print(a_temp,":-----------------------\n")
    #     model_ft = EfficientNet.from_name(net_name)
        
    #     # 加载网络权重
    #     net_weight = 'eff_weights/' + pth_map[net_name]
    #     # 修改模型
    #     state_dict = add_dict(torch.load(net_weight))
    #     model_ft.load_state_dict(state_dict)

    #     #对网络最后一层进行修改，即全链接层 1280->class_num
    #     num_ftrs = model_ft._fc.in_features
    #     # our methods
    #     model_ft._fc = nn.Linear(num_ftrs, (8-1)*2)
    #     # 消融实验
    #     # model_ft._fc = nn.Linear(num_ftrs,8)


    #     # 损失函数
    #     criterion = nn.CrossEntropyLoss().cuda()
    #     model_ft = model_ft.cuda()

    #     # 使用SGD用作训练 同类型的优化方法有ADAM等
    #     optimizer = optim.SGD((model_ft.parameters()), lr=lr,
    #                         momentum=momentum, weight_decay=0.0004)
    #     # 调用训练函数
    #     best_model_wts = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs,a = a_temp)

    #     #调用测试函数 输出模型评价指标
    #     print('-' * 10 + 'Test'+'-' * 10)

    #     model_ft = torch.load('mydata/model/efficientnet-b0('+str(a_temp)+').pth') #可单独使用测试函数
    #     # model_ft.load_state_dict(best_model_wts)
        
    #     criterion = nn.CrossEntropyLoss().cuda()
    #     test_model(model_ft, criterion,a = a_temp)
    #     a_temp  += 0.1

    a_temp = 0.7
    print(a_temp,":-----------------------\n")
    model_ft = EfficientNet.from_name(net_name)
        
    # 加载网络权重
    net_weight = 'eff_weights/' + pth_map[net_name]
    # 修改模型
    state_dict = add_dict(torch.load(net_weight))
    model_ft.load_state_dict(state_dict)

    #对网络最后一层进行修改，即全链接层 1280->class_num
    num_ftrs = model_ft._fc.in_features
    # our methods
    model_ft._fc = nn.Linear(num_ftrs, (8-1)*2)
    # 消融实验
    # model_ft._fc = nn.Linear(num_ftrs,8)


    # 损失函数
    criterion = nn.CrossEntropyLoss().cuda()
    model_ft = model_ft.cuda()

    # 使用SGD用作训练 同类型的优化方法有ADAM等
    optimizer = optim.SGD((model_ft.parameters()), lr=lr,
                        momentum=momentum, weight_decay=0.0004)
    # 调用训练函数
    best_model_wts = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs,a = a_temp)

    #调用测试函数 输出模型评价指标
    print('-' * 10 + 'Test'+'-' * 10)

    model_ft = torch.load('mydata/model/efficientnet-b0('+str(a_temp)+').pth') #可单独使用测试函数
    # model_ft.load_state_dict(best_model_wts)
        
    criterion = nn.CrossEntropyLoss().cuda()
    test_model(model_ft, criterion,a = a_temp)



   


            



