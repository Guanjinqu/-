import torch
import torch.nn as nn
import torchvision  #这里面直接加载MNIST数据的方法
import torchvision.transforms as transforms # 将数据转为Tensor
import torch.optim as optim 
import torch.utils.data.dataloader as dataloader 
from mpi4py import MPI
from fed_model import fed_fuc

train_set = torchvision.datasets.MNIST(
    root='./data', # 文件存储位置
    train=True,
    transform=transforms.ToTensor(),
    download=False
)

train_dataloader = dataloader.DataLoader(dataset=train_set,shuffle=True,batch_size=100)# 注意shuffle必须为True 否则数据集将会一样

'''
dataloader返回（images,labels）
其中，
images维度：[batch_size,1,28,28]
labels：[batch_size]，即图片对应的
'''

test_set = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor(),
    download=False
)

test_dataloader = dataloader.DataLoader(test_set,batch_size=100,shuffle=False) # dataset可以省

class NeuralNet(nn.Module):
    def __init__(self,in_num,h_num,out_num):
        super(NeuralNet,self).__init__()
        self.ln1 = nn.Linear(in_num,h_num)
        self.ln2 = nn.Linear(h_num,out_num)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        return self.ln2(self.relu(self.ln1(x)))

in_num = 784 # 输入维度
h_num = 500 # 隐藏层维度
out_num = 10 # 输出维度
epochs = 5 # 迭代次数
learning_rate = 0.001
#USE_CUDA = torch.cuda.is_available() # 定义是否可以使用cuda
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_nums = comm.Get_size()
name = "test2_10T"
f_name = name+"_loss"
g_name = name+"_acc"
if rank == 0:
    f = open(f_name,"w")
    g = open(g_name,"w")

model = NeuralNet(in_num,h_num,out_num) # 初始化模型

optimizer = optim.Adam(model.parameters(),lr=learning_rate) # 使用Adam
loss_fn = nn.CrossEntropyLoss() # 损失函数


fed_mode = True
loss_list = []
acc_list = []
def test():

    with torch.no_grad():
        total = 0
        correct = 0
        for (images,labels) in test_dataloader:
            images = images.reshape(-1,28*28)          
            result = model(images)
            prediction = torch.max(result, 1)[1] # 这里需要有[1]，因为它返回了概率还有标签
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
        acc_list.append(correct/total)
        print("The accuracy of total {} images: {}%".format(total, 100 * correct/total))


for e in range(epochs):
    for i,data in enumerate(train_dataloader):
        (images,labels) = data
        images = images.reshape(-1,28*28) # [batch_size,784]
        #if USE_CUDA:
        #    images = images.cuda() # 使用cuda
        #    labels = labels.cuda() # 使用cuda
            
        y_pred = model(images) # 预测
        loss = loss_fn(y_pred,labels) # 计算损失
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        n = e * i +1
        if n % 200 == 0 and fed_mode == True:
            #print("yes")
            fin_parameters = {}
            now_parameters = model.state_dict()
            recv_fin = comm.gather(now_parameters,root = 0)
            if rank == 0 :
                fin_parameters =  fed_fuc(recv_fin,mpi_nums)
            fin_parameters = comm.bcast(fin_parameters, root=0)    #需要注意的是 如果参数过大，需要调整buf大小
            model.load_state_dict(fin_parameters)
            model.eval()
            #print(model.state_dict())

        if n %100 == 0 and rank == 0:
            loss_list.append(loss.item())
            print(n,'loss:',loss.item())
            test()
        #if n %1000 == 0 and rank == 0:
        #    test()
        

if rank == 0 :
    f.write(str(loss_list))
    f.close()

    g.write(str(acc_list))
    g.close()