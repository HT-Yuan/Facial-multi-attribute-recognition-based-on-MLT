import matplotlib.pyplot as pl
import seaborn as sns
from sklearn import metrics
import numpy as np
# 相关库

def plot_matrix(y_true, y_pred, labels_name,gender_label,flag,axis_labels=None, name="1"):
    tr_list = []
    pr_list = []

    

    for tr,pr,gen in zip(y_true,y_pred, gender_label):
        if(gen == flag):
            tr_list.append(tr)
            pr_list.append(pr)


# 利用sklearn中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(tr_list, pr_list, labels=labels_name, sample_weight=None)  # 生成混淆矩阵 
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

# 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    if(flag == 2):
        pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
    else:
        pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('OrRd'))
    pl.colorbar()  # 绘制图例

# 图像标题
    # if title is not None:
    #     pl.title(title)
# 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    pl.xticks(num_local, axis_labels)  # 将标签印在x轴坐标上， 并倾斜45度
    pl.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    pl.ylabel('True label')
    pl.xlabel('Predicted label')

# # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 ) > 0:
                pl.text(j, i, format(int(cm[i][j] * 100), 'd') + '%',
                        ha="center", va="center",
                        color="white" if int(cm[i][j] * 100) > 85 else "black")  # 如果要更改颜色风格，需要同时更改此行
# 显示
    pl.savefig(name + ".jpg",dpi = 300 )

def age_sub(label,preds):
    result = {}
    for l,p in zip(label,preds):
        result[abs(l-p)] = result.get(abs(l - p), 0) + 1

    res = []
    for i in range(15):
        nums = 0
        if i > 0:
            nums = res[i - 1]
        nums += result.get(i,0)
        
        
        res.append(nums)
    res = [x / len(label) for x  in res]
    print("年龄累积和")
    print(res)

    # print(res)


    
    
def map_age_values(x):  
    if x in range(0,4):  
        return 0
    if x in range(4,13):  
        return 1
    if x in range(13,21):  
        return 2
    if x in range(21,31):  
        return 3
    if x in range(31,46):  
        return 4
    if x in range(46,61):  
        return 5
    if x > 60:  
        return 6
 

def plot_heatmap(pred_age,label_age,pred_gender,label_gender,pred_race,label_race,name_str,mode):
    C1 = np.zeros((7, 10))
    C_sum = np.zeros((7, 10))
    pred_age = list(map(map_age_values, pred_age)) 
    label_age = list(map(map_age_values, label_age)) 


    # for i in range(pred_age):

    for age, gender, race,pr_age,pr_gen,pr_rac in zip(label_age, label_gender,label_race,pred_age,pred_gender,pred_race ):

     

        if(gender):
            race += 5

        C_sum[age][race] += 1

        if(mode == 0):
            if(pr_age == age):
                C1[age][race] += 1

        if(mode == 1):
            if(pr_gen == gender):
                C1[age][race] += 1

        if(mode == 2): 
            if(pr_rac == race or pr_rac == race - 5):
                C1[age][race] += 1
        if(mode == 3): 
            if((pr_rac == race or pr_rac == race - 5) and pr_gen == gender and pr_age == age):
                C1[age][race] += 1
        # if(gender):
        #     if(pred == age ):
        #         C1[age][race + 5] += 1
        # else:
        #      C1[age][race] += 1



    C1 /= C_sum



    
    xtick=['White','Black','Asian','Indian','Others','White','Black','Asian','Indian','Others']
    ytick=['Baby','Child','Teenager','Young','Adult','Middle_Age','Senior']
     
   
    sns.heatmap(C1,fmt='g', cmap='RdBu_r',annot=False,cbar=True,xticklabels=xtick, yticklabels=ytick)


    pl.xticks(fontsize=7,rotation = 0)
    pl.yticks(fontsize=7)
    pl.title('Male                                                               Female',fontsize=7)

    

    pl.savefig(name_str + '.png',dpi = 300)
    

def plot_heatmap_test(name_str):
    C1 = np.zeros((7, 10))




        # if(gender):
        #     if(pred == age ):
        #         C1[age][race + 5] += 1
        # else:
        #      C1[age][race] += 1



    
    xtick=['White','Black','Asian','Indian','Others','White','Black','Asian','Indian','Others']
    ytick=['Baby','Child','Teenager','Young','Adult','Middle_Age','Senior']
     
   
    sns.heatmap(C1,fmt='g', cmap='RdBu_r',annot=False,cbar=True,xticklabels=xtick, yticklabels=ytick)


    pl.xticks(fontsize=7,rotation = 0)
    pl.yticks(fontsize=7)
    pl.title('Male                                                               Female',fontsize=7)

    

    pl.savefig(name_str + '.png',dpi = 300) 




if __name__ == "__main__":
    # y_true = [0,1]
    # y_pred = [0,1]
    # labels_name = [0,1]
    # plot_matrix(y_true, y_pred,labels_name )
    # plot_heatmap_test("tep")
    age_sub([0,1,2,3],[3,1,2,2])
