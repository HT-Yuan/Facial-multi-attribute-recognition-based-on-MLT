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
plt.figure(figsize = (10,6), dpi=100)
# plt.title("Anacomy of a figure",fontsize = 20,fontweight='heavy')

# 参数预设置
ax = plt.gca()
tick_step = 1
group_gap = 0.2 #组外间隔
bar_gap = 0.03

# raw = [0.96,0.913,0.882,0.941,0.963,0.942]
# sec = [0.985,0.964,0.926,0.953,0.978,0.899]
# proposed = [0.983,0.959,0.929,0.954,0.981,0.956]
# data = [raw, sec, proposed]

#raw1 = [0.725, 0.909]
raw1 = 2.89
raw2 = 74.53
raw3 = 1.63
raw4 = 0.86

data = [raw1, raw2, raw3,raw4]
group_num = len(data) #组数

group_width = tick_step - group_gap
# bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
bar_span = group_width / group_num
# bar_width为每个柱子的实际宽度
bar_width = bar_span - bar_gap


# ax.set_xlim(0,1)
# ax.set_ylim(50,100)
# yminorLocator   = MultipleLocator(2)
# ax.yaxis.set_minor_locator(yminorLocator)
# ax.set_xlabel("Class",fontsize = 16)
ax.set_ylabel("Time Cost (ms)",fontsize = 16)
plt.grid(linewidth = 0.5,alpha = 0.7,linestyle = (2,(20,5)),axis = 'y')
colors = ['green','purple','yellow','blue']
labels = ['Jetson TX2','ARM A9','GTX 970M','GTX 1060']
i = 0
for index, y in enumerate(data):
        ax.bar(index*bar_span, y, bar_width,color=colors[i],edgecolor='black',label=labels[i])


        # for qidian,num in zip(index*bar_span,y):
        plt.text(index*bar_span, y,'%.2f'%y, ha = 'center',fontsize=13,rotation=0)
        i+=1


# ticks = x + (group_width - bar_span) / 2
# plt.xticks(ticks, label,fontsize=13)
#plt.axis('tight')
# ax.legend()
plt.xticks([0,0.2,0.4,0.6], labels)

plt.savefig("4-1.jpg",dpi=1200,format='jpg')
# plt.show()
