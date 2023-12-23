## UTK_FACE数据集 年龄分布统计图
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess

# sns.set(font='SimHei',style='darkgrid')

# 测试集 + 验证集

# test_list = []

def count_file(directory = "."):
    test_list = []
    for i in range(0,117):
        cmd = f"find  UTK_Face/train UTK_Face/train -type f -name \"{i}_*_*_*.jpg\" | wc -l"
        resutlt = subprocess.run(cmd,stdout=subprocess.PIPE,shell=True,text=True)
        test_list.append(int(resutlt.stdout.strip()))
    return test_list


# print(count_file())





# total_num2 = [86.1,88.4,87.0,71.2,80.3,63.3,33.8,49.0]
# # total_num1 = [94.0,90.8,86.3,80.5,88.2,70.9,60.1,74.5]
# total_num1 = [90.08,91.9,88.28,77.18,85.52,74.61,60,77.55]
line_data = count_file()
print(line_data)
# index_name = []
# for i in range(0,117):
#     index_name.append(str(i))

# # index_name = ['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60-100']
# # color_name = ['blue','blue','blue','blue','darkorchid','darkorchid','darkorchid','orange']



# # # bar1_list = plt.bar(index_name,total_num1,label = 'Proposed I',color = color_name)
# # # bar2_list = plt.bar(index_name,-np.array(total_num2),label = 'E-B0',color = color_name,alpha = 0.5)
# plt.plot(index_name,line_data,color = 'r',marker = 'o',markersize = 3)
# # # plt.fill_between(index_name,line_data,0,alpha = 0.6,color = 'red')


# # # 上面的柱状图加上标签
# # for bar1 in bar1_list:
# #     plt.text(bar1.get_x() + bar1.get_width()/2,bar1.get_height() ,bar1.get_height(),ha = 'center')

# # # 下面的柱状图加上标签
# # for bar2 in bar2_list:
# #     plt.text(bar2.get_x() + bar2.get_width()/2,bar2.get_height()-5,-bar2.get_height(),ha = 'center')

# # for bar2,da in zip(bar2_list,line_data):
# #     plt.text(bar2.get_x() + bar2.get_width()/2,round(da,2),round(da,2),ha = 'center',color = 'white')

# # # plt.title('')
# plt.xlabel('Age')
# plt.ylabel('Num')
# plt.xticks(range(0,120,10))
# plt.grid(True)

# # plt.legend(loc=4)

# plt.savefig("utk_data_analyse.jpg",dpi=200)
