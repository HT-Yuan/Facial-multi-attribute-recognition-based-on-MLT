import numpy as np
import matplotlib.pyplot as plt



# def data_read(dir_path):
#     with open(dir_path, "r") as f:
#         raw_data = f.read()
#         data = raw_data.split(" ")   # [-1:1]是为了去除文件中的前后中括号"[]"

#     return np.asfarray(data, float)
def plot_results0(dir_path, num_epochs): # 绘制 基于不确定性图 的损失图

    f = open(dir_path, "r",encoding='utf-8')
    name = str(dir_path).split('/')[-1]
    name = name.split(".")[0]
    line = f.readline() # 读取第一行
    epoches = []
    l1 = []
    l2 = []
    l3 = []
    acc1 = []
    acc2 = []
    index = 0
    while line:
        if(index >= num_epochs):
            break
        data = line.split()
        # data = np.array(data,dtype=float)
        # data = map(float,data)
        # print(data[0])
        # txt_data = eval(line) # 可将字符串变为元组
        epoches.append(int(data[0])) # 列表增加
        l1.append(float(data[1])/0.7)
        l2.append(float(data[2])/0.3)
        # l3.append(float(data[3])*100)
        acc1.append(float(data[3])*100)
        acc2.append(float(data[4])*100)
        line = f.readline() # 读取下一行
        index +=1
    
    f.close()
    print(epoches)

    plt.figure(figsize=(8, 8))
    plt.xlabel('iters')    # x轴标签
    plt.ylabel('loss')     # y轴标签
    plt.plot(epoches, l1, linewidth=1, linestyle="solid", label="train loss1",color="blue")
    plt.plot(epoches, l2, linewidth=1, linestyle="solid", label="train loss2",color="red")
    # plt.plot(epoches, l3, linewidth=1, linestyle="solid", label="train loss3",color="y")
    plt.legend()
    plt.title('Loss curve')
  
    plt.savefig(f'./{name}.png')

def plot_results(dir_path, num_epochs,name, dirpath_2): # 绘制 基于不确定性图 的损失图

    f = open(dir_path, "r",encoding='utf-8')
    line = f.readline() # 读取第一行
    epoches = []
    l1 = []
    l2 = []
    l3 = []
    acc1 = []
    acc2 = []
    p1 = []
    p2 = []
    p3 = []
    index = 0
    while line:
        if(index >= num_epochs):
            break
        data = line.split()
        # data = np.array(data,dtype=float)
        # data = map(float,data)
        # print(data[0])
        # txt_data = eval(line) # 可将字符串变为元组
        epoches.append(int(data[0])) # 列表增加
        l1.append(float(data[1]))
        l2.append(float(data[2]))
        l3.append(float(data[3]))
        # acc1.append(float(data[4])*100)
        # acc2.append(float(data[5])*100)
        p1.append(float(data[-3]))
        p2.append(float(data[-2]))
        p3.append(float(data[-1]))
        # p1.append(float(data[6]))
        # p2.append(float(data[7]))
        line = f.readline() # 读取下一行
        index +=1
    
    f.close()

    f = open(dirpath_2, "r",encoding='utf-8')
    line = f.readline() # 读取第一行
    epoches = []
 

    p1_ad = []
    p2_ad = []
    index = 0
    while line:
        if(index >= num_epochs):
            break
        data = line.split()

        epoches.append(int(data[0])) # 列表增加
    
        p1_ad.append(float(data[6]))
        p2_ad.append(float(data[7]))
        line = f.readline() # 读取下一行
        index +=1
    
    f.close()


    # print(np.sqrt(np.exp(p1)))
    # fig,axs = plt.subplots(1, 1, figsize=(6, 10))

  
    # axs[1].set_xlabel('epoch', fontsize=12)
    # axs[1].set_ylabel('weight', fontsize=12)
    

    plt.figure(figsize=(10, 8))
    plt.xlabel('Epochs',fontsize=12)    # x轴标签
    plt.ylabel('Weight',fontsize=12)     # y轴标签
    
    # weight1 = 1/(np.exp(p1))
    # weight2 = 1/(np.exp(p2))
    # weight3 = 1/(np.exp(p3))
    
    # plt.plot(epoches, weight1/(weight1+weight2+weight3), linewidth=3, linestyle="solid", label="β_age",color="blue")
    # plt.plot(epoches, weight2/(weight1+weight2+weight3), linewidth=3, linestyle="solid", label="β_gender",color="red")
    # plt.plot(epoches, weight3/(weight1+weight2+weight3), linewidth=3, linestyle="solid", label="β_race",color="purple")
    # plt.legend()
   

    # axs[1].plot(epoches, weight1/(weight1+weight2+weight3), linewidth=3, linestyle="solid", label="β_age",color="blue")
    # axs[1].plot(epoches, weight2/(weight1+weight2+weight3), linewidth=3, linestyle="solid", label="β_gender",color="red")
    # axs[1].plot(epoches, weight3/(weight1+weight2+weight3), linewidth=3, linestyle="solid", label="β_race",color="purple")
    # axs[1].legend()
    # axs[1].set_title("UTK Face")

    # print(weight1/(weight1+weight2 +weight3 ))
    # print(weight2/(weight1+weight2 +weight3 ))
    # print(weight3/(weight1+weight2 +weight3 ))


    # axs[0].set_xlabel('epoch', fontsize=12)
    # axs[0].set_ylabel('weight', fontsize=12)
    
    weight1 = 1/(np.exp(p1_ad))
    weight2 = 1/(np.exp(p2_ad))
    # axs[0].plot(epoches, weight1/(weight1+weight2), linewidth=3, linestyle="solid", label="β_age",color="blue")
    # axs[0].plot(epoches, weight2/(weight1+weight2), linewidth=3, linestyle="solid", label="β_gender",color="red")
    # axs[0].legend()
    # axs[0].set_title("Adience")

    # print(weight1/(weight1+weight2 ))
    # print(weight2/(weight1+weight2 ))

    plt.plot(epoches, weight1/(weight1+weight2), linewidth=3, linestyle="solid", label="β_age",color="blue")
    plt.plot(epoches, weight2/(weight1+weight2), linewidth=3, linestyle="solid", label="β_gender",color="red")
    
    # plt.plot(epoches, weight1/(weight1+weight2 +weight3 ), linewidth=3, linestyle="solid", label="β_age",color="blue")
    # plt.plot(epoches, weight2/(weight1+weight2 +weight3 ), linewidth=3, linestyle="solid", label="β_gender",color="red")
    # plt.plot(epoches, weight3/(weight1+weight2 +weight3 ), linewidth=3, linestyle="solid", label="β_race",color="purple")
    plt.legend()
    # plt.title('Loss curve')
    # plt.show()
    plt.savefig(name + '.png',dpi=200)

def plot_ac(name): # 绘制 基于不确定性图 的损失图

    mlt = [0.14478323999170298, 0.29330014519809167, 0.41008089607965154, 0.49346608587429996, 0.5774735532047293, 0.6490354698195395, 0.7066998548019083, 0.7465256170918897, 0.7871810827629122, 0.8257622899813317, 0.855216760008297, 0.8784484546774528, 0.8975316324414022, 0.9122588674548848, 0.9267786766231072]
    order = [0.1281891723708774, 0.2671644886952914, 0.3820784069695084, 0.47085666874092513, 0.5457373988799005, 0.6172993154947106, 0.676000829703381, 0.7201825347438291, 0.7649865173200581, 0.8033602986932171, 0.8323999170296619, 0.8574984443061605, 0.8772038996058908, 0.8946276706077577, 0.9089400539307197]
    hard = [0.1557768097904999, 0.2767060775772661, 0.3833229620410703, 0.4596556730968679, 0.527276498651732, 0.5884671230035262, 0.6409458618543871, 0.6845052893590542, 0.7268201617921594, 0.7689276083800042, 0.8019083177763949, 0.8267994192076332, 0.8473345778884049, 0.8630989421281892, 0.8796930097490148]
    plt.figure(figsize=(12, 8))
    plt.xlabel('deviation')    # x轴标签
    plt.ylabel('cs(%)')     # y轴标签
    # plt.plot(epoches, l1, linewidth=1, linestyle="solid", label="train loss1",color="blue")
    # plt.plot(epoches, l2, linewidth=1, linestyle="solid", label="train loss2",color="red")
    # plt.plot(epoches, l3, linewidth=1, linestyle="solid", label="train loss3",color="y")

    # weight3 = np.sqrt(np.exp(p3))
    # print(weight1/(weight1+weight2 +weight3 ),weight2/(weight1+weight2 +weight3 ), weight3/(weight1+weight2 +weight3 ))
    plt.plot(range(0,15), mlt, linewidth=4, linestyle="solid", label="Proposed",color="red")
    plt.scatter(range(0,15), mlt, s=40, color="r")
    plt.plot(range(0,15), hard, linewidth=4, linestyle="solid", label="Proposed*",color="y")
    plt.scatter(range(0,15), hard, s=40, color="y")
    plt.plot(range(0,15), order, linewidth=4, linestyle="solid", label="Proposed**",color="b")
    plt.scatter(range(0,15), order, s=40, color="b")
    
    
    # plt.plot(epoches, weight1/(weight1+weight2 +weight3 ), linewidth=3, linestyle="solid", label="β_age",color="blue")
    # plt.plot(epoches, weight2/(weight1+weight2 +weight3 ), linewidth=3, linestyle="solid", label="β_gender",color="red")
    # plt.plot(epoches, weight3/(weight1+weight2 +weight3 ), linewidth=3, linestyle="solid", label="β_race",color="purple")
    plt.legend()
    # plt.title('Loss curve')
    plt.show()
    plt.savefig(name + '.png')

def plot_mat(name): 
    matrix = np.empty((10, 10))
    print(matrix)
    for i in range(matrix.shape[0]):  
        for j in range(matrix.shape[1]):  
            matrix[i, j] = np.abs(j - i) 
   
 



    plt.figure(figsize=(8,6))
    ax = plt.imshow(matrix)
    plt.colorbar(ax.colorbar, fraction=0.025)
    plt.axis('tight')
    plt.show()
    plt.savefig(name + '.png')
if __name__ == "__main__":
    #plot_results("/media/omnisky/a6aeaf75-c964-444d-b0ed-d248f1370cd5/yhq/mlt-Project/csdn_temp/mydata/answer/(mlt_adience_EEE).txt",80,"adience_un16")
    plot_results("/media/omnisky/a6aeaf75-c964-444d-b0ed-d248f1370cd5/yhq/mlt-Project/csdn_temp/UTK_Face/answer/utk_mlt_e.txt",80,"adience","/media/omnisky/a6aeaf75-c964-444d-b0ed-d248f1370cd5/yhq/mlt-Project/csdn_temp/mydata/answer/(mlt_adience_EEE).txt")
    # plot_ac("ac")
    # plot_mat("mat")
    # plot_results0("/media/omnisky/a6aeaf75-c964-444d-b0ed-d248f1370cd5/yhq/mlt-Project/csdn_temp/mydata/answer/(order_adience).txt",80)
    # plot_results0("/media/omnisky/a6aeaf75-c964-444d-b0ed-d248f1370cd5/yhq/mlt-Project/csdn_temp/mydata/answer/(order_adience).txt",80)
    # print(x)