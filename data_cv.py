import cv2
import os
from random import sample
# from sklearn.model_selection import train_test_split
# from shutil import copyfile

# sets = ["test","train"]
# class_name = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
# a = 0
# def detect(filename,dirname):
#     global a
#     # cv2级联分类器CascadeClassifier,xml文件为训练数据
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#     # 读取图片
#     img = cv2.imread(filename)
#     # 转灰度图
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # 进行人脸检测
#     faces = face_cascade.detectMultiScale(img, 1.3, 5)
#     # print (faces[1])
#     # 绘制人脸矩形框
#     for (x, y, w, h) in faces:
#         #img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         img = img[y:y+h,x:x+w]
#         cv2.resize(img,(224,224))
#         cv2.imwrite(dirname, img)
#         a+=1
#         print(a)
#         break


# # for image_set in sets:
# #     for class_n in class_name:
# #         for i in os.listdir(os.path.join('face_data',image_set,class_n)):
# #             souce_name = os.path.join('my_data',image_set,class_n,i)
# #             dir_name = os.path.join('face_data',image_set,class_n)
# #             if not os.path.exists(dir_name):
# #                 os.makedirs(dir_name)
# #             dir_name =  os.path.join(dir_name,i)
# #             detect(souce_name,dir_name)

# # for image_set in sets:
# #     for class_n in class_name:
# #         print(class_n + 'start')
# #         data = os.listdir(os.path.join('face_data',image_set,class_n))
# #         train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
# #         for i in train_df:
# #             souce_name = os.path.join('face_data',image_set,class_n,i)
# #             dir_name = os.path.join('face_data2','train',class_n)
# #             if not os.path.exists(dir_name):
# #                 os.makedirs(dir_name)
# #             dir_name =  os.path.join(dir_name,i)
# #             copyfile(souce_name,dir_name)
# #         for i in test_df:
# #             souce_name = os.path.join('face_data',image_set,class_n,i)
# #             dir_name = os.path.join('face_data2','test',class_n)
# #             if not os.path.exists(dir_name):
# #                 os.makedirs(dir_name)
# #             dir_name =  os.path.join(dir_name,i)
# #             copyfile(souce_name,dir_name)
# #         print(class_n + 'done')
# m = {}
# for class_n in class_name:
#     data = os.listdir(os.path.join('face_data2','test',class_n))
#     m[class_n] = len(data)
# print(m)

def detect(filename,dirname):
    # cv2级联分类器CascadeClassifier,xml文件为训练数据
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # 读取图片
    img = cv2.imread(filename)
    # 转灰度图
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行人脸检测
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    # print (faces[1])
    # 绘制人脸矩形框
    for (x, y, w, h) in faces:
        #img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        img = img[y:y+h,x:x+w]
        cv2.resize(img,(224,224))
        cv2.imwrite(dirname, img)

        break

if __name__=='__main__':
    # train_size = len(os.listdir('LFW_Data/train'))
    # test_size = len(os.listdir('LFW_Data/test'))
    # print(train_size,test_size)
    # dstpath = 'Predict'

    # img = cv2.resize(img,(224,224))
    # cv2.imwrite('Predict/3860.jpg', img)
    

    # files= os.listdir(dstpath)
    # for file in files:
    #     print(file)

    #     dstpath_temp = os.path.join(dstpath,file)
    #     picture_name = os.listdir(dstpath_temp)
    #     for name in picture_name:
    #         img = cv2.imread(os.path.join(dstpath_temp,name))
    #         print(img.shape)
    #         break
    #         print(os.path.join(srcpath_temp,name))


        # picture_name = os.listdir(srcpath_temp)
        # for name in picture_name:
        #     # print(os.path.join(srcpath_temp,name))
            
        #     detect(os.path.join(srcpath_temp,name),os.path.join(dstpath_temp,name))

    # srcpath = 'LFW_Data/test'
    # dstpath = 'LFW_noface/test'
    # if not os.path.isdir(dstpath):
    #     os.mkdir(dstpath) 

    # files= os.listdir(srcpath)
    # for file in files:
    #     srcpath_temp = os.path.join(srcpath,file)
    #     dstpath_temp = os.path.join(dstpath,file)
    #     if not os.path.isdir(dstpath_temp):
    #         os.mkdir(dstpath_temp)
    #         print(dstpath_temp)

    #     picture_name = os.listdir(srcpath_temp)
    #     for name in picture_name:
    #         # print(os.path.join(srcpath_temp,name))
    #         print(dstpath_temp)
    #         detect(os.path.join(srcpath_temp,name),os.path.join(dstpath_temp,name))
    
    # # 将all中的数据放到test
    # dst_path = 'LFW_noface1/all'
    # files= os.listdir(dst_path)
    # for file in files:
    #     srcpath_temp = os.path.join(dst_path,file)
    #     dstpath_temp = os.path.join('LFW_noface1/test',file)
    #     picture_name = os.listdir(srcpath_temp)
    #     if(len(picture_name) >=10):
    #         test_sampe = sample(picture_name, int(len(picture_name)*0.1))
    #         # print(test_sampe)
    #         for name in test_sampe:
    #             os.rename(os.path.join(srcpath_temp,name),os.path.join(dstpath_temp,name))
    # 测试双方数据
    dst0_path = 'LFW_noface/train'
    dst1_path = 'LFW_noface/tarin'
    files= os.listdir(dst0_path)
    test_num = 0
    for file in files:
        srcpath_temp = os.path.join(dst0_path,file)
        picture_name = os.listdir(srcpath_temp)
        print(file,len(picture_name))
        test_num += len(picture_name)
    print(test_num)



    # train_num = {}
    # test_num = {}

    # 将test/train汇总成all
    # list_dir = ['test','train']
    # source_dir = 'LFW_noface1'
    # dist_dir = 'LFW_noface1/all'
    # for dir in list_dir:
    #     source_path = os.path.join(source_dir,dir)

    #     files= os.listdir(source_path)
    #     for file in files:
    #         dist_dirfile = os.path.join(dist_dir,file)
    #         src_dirfile = os.path.join(source_path,file)
    #         print(src_dirfile)

    #         if not os.path.isdir(dist_dirfile):
    #             os.mkdir(dist_dirfile)

    #         picture_name = os.listdir(os.path.join(source_path,file))
    #         for name in picture_name:
    #             os.rename(os.path.join(src_dirfile,name),os.path.join(dist_dirfile,name))

    # print(train_num)
    # print(test_num)
    # all_num = {}
    # files= os.listdir(source_path)
    # for file in files:
    #     if (train_num[file]+test_num[file] != 0):
    #         all_num[file] = test_num[file]/(train_num[file]+test_num[file])
    #     else:
    #         all_num[file] = 0
    # print(all_num)