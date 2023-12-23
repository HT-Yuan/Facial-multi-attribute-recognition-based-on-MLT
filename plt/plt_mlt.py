# import numpy as np
# import matplotlib.pyplot as plt

# size = 5
# x = np.arange(size)
# a = np.random.random(size)
# b = np.random.random(size)
# c = np.random.random(size)

# total_width, n = 0.8, 3
# width = total_width / n
# x = x - (total_width - width) / 2

# plt.bar(x, a,  width=width, label='a')
# plt.bar(x + width, b, width=width, label='b')
# plt.bar(x + 2 * width, c, width=width, label='c')
# plt.legend()
# plt.show()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# plt.rc('font',family='Avenir') 

# adience
# plt.figure(figsize = (10,6), dpi=100)
# # plt.title("Anacomy of a figure",fontsize = 20,fontweight='heavy')

# # 参数预设置
# ax = plt.gca()
# tick_step = 1
# group_gap = 0.2 #组外间隔
# bar_gap = 0.03
# label = ['Age', 'Gender']
# x = np.arange(len(label)) * tick_step
# # raw = [0.96,0.913,0.882,0.941,0.963,0.942]
# # sec = [0.985,0.964,0.926,0.953,0.978,0.899]
# # proposed = [0.983,0.959,0.929,0.954,0.981,0.956]
# # data = [raw, sec, proposed]

# #raw1 = [0.725, 0.909]
# raw1 = [66.1, 81.8]
# raw2 = [80.4, 90.0]
# raw3 = [82.78, 94.08]
# raw4 = [83.7, 94.13]
# raw5 = [86.59, 95.75]
# data = [raw1, raw2, raw3,raw4,raw5]
# group_num = len(data) #组数

# group_width = tick_step - group_gap
# # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
# bar_span = group_width / group_num
# # bar_width为每个柱子的实际宽度
# bar_width = bar_span - bar_gap


# # ax.set_xlim(0,1)
# ax.set_ylim(50,100)
# yminorLocator   = MultipleLocator(2)
# ax.yaxis.set_minor_locator(yminorLocator)
# # ax.set_xlabel("Class",fontsize = 16)
# ax.set_ylabel("Accuracy Values (%)",fontsize = 16)
# plt.grid(linewidth = 0.5,alpha = 0.7,linestyle = (2,(20,5)),axis = 'y')
# colors = ['green','purple','yellow','blue','red']
# labels = ['GRA-Net','AFA-Net','Proposed*','Proposed**','Proposed']
# i = 0
# for index, y in enumerate(data):
#         ax.bar(x + index*bar_span, y, bar_width,color=colors[i],edgecolor='black',label=labels[i])

#         for qidian,num in zip(x + index*bar_span,y):
#             plt.text(qidian, num,'%.2f'%num, ha = 'center',fontsize=13,rotation=0)
#         i+=1


# ticks = x + (group_width - bar_span) / 2
# plt.xticks(ticks, label,fontsize=13)
# #plt.axis('tight')
# ax.legend()
# # plt.xticks([0,1,2,3,4,5], label)

# plt.savefig("3-8.jpg",dpi=1200,format='jpg')

# data = [first, second, third]


# def create_multi_bars(labels, datas, tick_step=1, group_gap=0.2, bar_gap=0.01):
#     '''
#     labels : x轴坐标标签序列
#     datas ：数据集，二维列表，要求列表每个元素的长度必须与labels的长度一致
#     tick_step ：默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
#     group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
#     bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
#     '''
#     # x为每组柱子x轴的基准位置 [0,1,2,3,4,5]
#     x = np.arange(len(labels)) * tick_step
 
#     # group_num为数据的组数，即每组柱子的柱子个数
#     group_num = len(datas)
#     # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
#     group_width = tick_step - group_gap
#     print(group_width)
#     # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
#     bar_span = group_width / group_num
#     # bar_width为每个柱子的实际宽度
#     bar_width = bar_span - bar_gap
#     plt.figure(dpi=60,figsize=(10,6))
#     # 绘制柱子
#     for index, y in enumerate(datas):
#         plt.bar(x + index*bar_span, y, bar_width)
#     plt.ylabel('map@0.5')
#     plt.xlabel('class')
#     # plt.title('multi datasets')
#     # ticks为新x轴刻度标签位置，即每组柱子x轴上的中心位置
#     ticks = x + (group_width - bar_span) / 2
#     plt.xticks(ticks, labels)
#     plt.show()

# create_multi_bars(label, data[:3])

plt.figure(figsize = (10,6), dpi=100)
# plt.title("Anacomy of a figure",fontsize = 20,fontweight='heavy')

# 参数预设置
ax = plt.gca()
tick_step = 1
group_gap = 0.2 #组外间隔
bar_gap = 0.03
label = ['Age', 'Gender','Race']
x = np.arange(len(label)) * tick_step
# raw = [0.96,0.913,0.882,0.941,0.963,0.942]
# sec = [0.985,0.964,0.926,0.953,0.978,0.899]
# proposed = [0.983,0.959,0.929,0.954,0.981,0.956]
# data = [raw, sec, proposed]

#raw1 = [0.725, 0.909]
raw1 = [61.78, 90.63,78.49]
raw2 = [62.48, 90.7,78.8]
raw3 = [64.74, 90.91,79.78]

data = [raw1, raw2, raw3]
group_num = len(data) #组数

group_width = tick_step - group_gap
# bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
bar_span = group_width / group_num
# bar_width为每个柱子的实际宽度
bar_width = bar_span - bar_gap


# ax.set_xlim(0,1)
ax.set_ylim(20,100)
yminorLocator   = MultipleLocator(2)
ax.yaxis.set_minor_locator(yminorLocator)
# ax.set_xlabel("Class",fontsize = 16)
ax.set_ylabel("Accuracy Values (%)",fontsize = 16)
plt.grid(linewidth = 0.5,alpha = 0.7,linestyle = (2,(20,5)),axis = 'y')
colors = ['yellow','blue','red']
labels = ['Proposed*','Proposed**','Proposed']

i = 0
for index, y in enumerate(data):
        ax.bar(x + index*bar_span, y, bar_width,color=colors[i],edgecolor='black',label=labels[i])

        for qidian,num in zip(x + index*bar_span,y):
            plt.text(qidian, num,'%.2f'%num, ha = 'center',fontsize=13,rotation=0)
        i+=1


ticks = x + (group_width - bar_span) / 2
plt.xticks(ticks, label,fontsize=13)
#plt.axis('tight')
ax.legend()
# plt.xticks([0,1,2,3,4,5], label)

plt.savefig("3-9.jpg",dpi=1200,format='jpg')