import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# adience数据集 年龄前后分布对比

# sns.set(font='SimHei',style='darkgrid')

# total_num2 = [86.1,88.4,87.0,71.2,80.3,63.3,33.8,49.0]
# # total_num1 = [94.0,90.8,86.3,80.5,88.2,70.9,60.1,74.5]
# total_num1 = [90.08,91.9,88.28,77.18,85.52,74.61,60,77.55]


# adience数据集 年龄前后分布对比
total_num2 = [82.61,79.37,72.82,63.7,78.59,64.56,52.04,69.47]
# total_num1 = [94.0,90.8,86.3,80.5,88.2,70.9,60.1,74.5]
total_num1 = [89.16,90.42,85.87,75.26,85.92,75.93,62.67,80.28]
line_data = [a - b for a,b in zip(total_num1,total_num2)]
index_name = ['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60-100']
color_name = ['red','blue','blue','blue','darkorchid','darkorchid','darkorchid','orange']


plt.figure(figsize=(10, 8))
bar1_list = plt.bar(index_name,total_num1,label = 'Proposed*',color = color_name)
bar2_list = plt.bar(index_name,-np.array(total_num2),label = 'E-B0',color = color_name,alpha = 0.4)
# plt.plot(index_name,line_data,color = 'r')
plt.fill_between(index_name,line_data,0,alpha = 0.8,color = 'pink')


# 上面的柱状图加上标签
for bar1 in bar1_list:
    plt.text(bar1.get_x() + bar1.get_width()/2,bar1.get_height() ,bar1.get_height(),ha = 'center')

# 下面的柱状图加上标签
for bar2 in bar2_list:
    plt.text(bar2.get_x() + bar2.get_width()/2,bar2.get_height()-5,-bar2.get_height(),ha = 'center')

for bar2,da in zip(bar2_list,line_data):
    plt.text(bar2.get_x() + bar2.get_width()/2,round(da,2),round(da,2),ha = 'center',color = 'white')

# plt.title('')
plt.xlabel('Age Group')
plt.ylabel('F1 %')
# plt.yticks([-40,-20,0,20,40],[40,20,0,20,40])

plt.legend(loc=4)

plt.savefig("Adience_F1.jpg",dpi=200)


####################################UTK##################################
# total_num2 = [86.71,58.24,38.91,65.49,47.01,47.64,67.11]
# # total_num1 = [94.0,90.8,86.3,80.5,88.2,70.9,60.1,74.5]
# total_num1 = [89.79,59.21,37.93,65.51,47.06,49.79,67.14]
# line_data = [a - b for a,b in zip(total_num1,total_num2)]
# index_name = ['Baby','Child','Teenager','Yong','Adult','Middle_Age','Senior',]
# color_name = ['red','blue','blue','blue','darkorchid','darkorchid','darkorchid']
# widths = [0.5, 9*0.08, 7*0.08, 11*0.08,15*0.06,15*0.06,30*0.03]


# plt.figure(figsize=(12, 8))
# bar1_list = plt.bar(index_name,total_num1,label = 'Proposed*',color = color_name,width=widths)
# bar2_list = plt.bar(index_name,-np.array(total_num2),label = 'E-B0',color = color_name,alpha = 0.4,width=widths)
# # plt.plot(index_name,line_data,color = 'r')
# plt.fill_between(index_name,line_data,0,alpha = 0.8,color = 'pink')


# # 上面的柱状图加上标签
# for bar1 in bar1_list:
#     plt.text(bar1.get_x() + bar1.get_width()/2,bar1.get_height() ,bar1.get_height(),ha = 'center')

# # 下面的柱状图加上标签
# for bar2 in bar2_list:
#     plt.text(bar2.get_x() + bar2.get_width()/2,bar2.get_height()-5,-bar2.get_height(),ha = 'center')

# for bar2,da in zip(bar2_list,line_data):
#     plt.text(bar2.get_x() + bar2.get_width()/2,round(da,2),round(da,2),ha = 'center',color = 'white')

# # plt.title('')
# plt.xlabel('Age Group')
# plt.ylabel('F1 %')
# # plt.yticks([-40,-20,0,20,40],[40,20,0,20,40])

# plt.legend(loc=4)

# plt.savefig("UTK Face_F1.jpg",dpi=200)
