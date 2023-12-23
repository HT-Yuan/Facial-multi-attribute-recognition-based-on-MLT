# 数据集的划分


import os
import random
import shutil
 
 
 
 
def moveFile(input1,save1):
    pathDir = os.listdir(input1)  # 取图片的原始路径
    random.seed(1)
    filenumber = len(pathDir)  # 原文件个数
    rate = 0.2  # 抽取的验证集的比例，占总数据的多少
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取需要数量的样本图片
    print(sample)
    list_len=len(sample)
    print(list_len)
    list=[]
    for i in range(len(sample)):
        list.append(sample[i])
    # print(list)
    for flie_name in list:
        path_img=os.path.join(input1,flie_name)
        print(path_img, "->" , save1)
        shutil.move(path_img,save1)
        
 
if __name__ == '__main__':
    input_path = './UTK_Face/train'
    output_path = './UTK_Face/test'
    moveFile(input_path,output_path)
