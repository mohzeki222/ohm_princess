import numpy as np
import matplotlib.pyplot as plt

import chainer.optimizers as Opt
import chainer.functions as F
import chainer.links as L
import chainer
import chainer.serializers as ser

from chainer import Variable,Chain,config,cuda
from tqdm import tqdm

import os
import PIL.Image as im        

import urllib.error
import urllib.request

def download_image(url, dst_path):
    try:
        data = urllib.request.urlopen(url).read()
        with open(dst_path, mode="wb") as f:
            f.write(data)
    except urllib.error.URLError as error:
        print(error)
        
def download_figs():
    
    url_list = []
    for i in range(19):
        url = "https://github.com/mohzeki222/ohm_princess/blob/master/notes/princess_fig/kisaki{:02}.jpg".format(i)
        url_list.append(url)
    download_dir = "princess_fig"
    for url in url_list:
        filename = os.path.basename(url)
        dst_path = os.path.join(download_dir, filename)
        download_image(url, dst_path)
        
    url_list = []
    for i in range(8):
        url = "https://github.com/mohzeki222/ohm_princess/blob/master/notes/white_fig/sira{:02}.jpg".format(i)
        url_list.append(url)
    download_dir = "white_fig"
    for url in url_list:
        filename = os.path.basename(url)
        dst_path = os.path.join(download_dir, filename)
        download_image(url, dst_path)

class ResBlock(Chain):
    def __init__(self, ch, bn=True):
        layers = {}
        layers['conv1'] = L.Convolution2D(ch,ch,3,1,1)
        layers['conv2'] = L.Convolution2D(ch,ch,3,1,1)
        layers['bnorm1'] = L.BatchNormalization(ch)
        layers['bnorm2'] = L.BatchNormalization(ch)
        super().__init__(**layers)
        
    def __call__(self,x):
        h = self.conv1(x)
        if self.bn == True:
            h = self.bnorm1(h)
        h = F.relu(h)
        h = self.conv2(h)
        if self.bn == True:
            h = self.bnorm2(h)
        h = h +x
        h = F.relu(h)

        return h
    
    
class Bottleneck(Chain):
    def __init__(self, ch, bn=True):
        layers = {}
        layers['conv1'] = L.Convolution2D(ch,ch,1,1,0)
        layers['conv2'] = L.Convolution2D(ch,ch,3,1,1)
        layers['conv3'] = L.Convolution2D(ch,ch,1,1,0)
        layers['bnorm1'] = L.BatchNormalization(ch)
        layers['bnorm2'] = L.BatchNormalization(ch)
        layers['bnorm3'] = L.BatchNormalization(ch)
        super().__init__(**layers)
        
    def __call__(self,x):
        h = self.conv1(x)
        if self.bn == True:
            h = self.bnorm1(h)
        h = F.relu(h)
        h = self.conv2(h)
        if self.bn == True:
            h = self.bnorm2(h)
        h = F.relu(h)
        h = self.conv3(h)
        if self.bn == True:
            h = self.bnorm3(h)
        h = h +x
        h = F.relu(h)
        return h
    
    
class CBR(Chain):
    def __init__(self, ch_in, ch_out, sample='down', bn=True, act=F.relu, drop=False):
        self.bn = bn
        self.act = act
        self.drop = drop

        layers = {}
        if sample=='down':
            layers['conv'] = L.Convolution2D(ch_in, ch_out, 4, 2, 1)
        else:
            layers['conv'] = L.Deconvolution2D(ch_in, ch_out, 4, 2, 1)
        if bn:
            layers['bnorm'] = L.BatchNormalization(ch_out)
        super().__init__(**layers)
        
    def __call__(self, x):
        h = self.conv(x)
        if self.bn == 1:
            h = self.bnorm(h)
        if self.drop == 1:
            h = F.dropout(h)
        h = self.act(h)
        return h
    
    
class CNN(Chain):
    def __init__(self, ch_in,ch_out,ksize=3,stride=2,pad=1,pooling=True):
        self.pooling = pooling
        layers = {}
        layers['conv1'] = L.Convolution2D(ch_in,ch_out,ksize=ksize,stride=stride,pad=pad)
        layers['bnorm1'] = L.BatchNormalization(ch_out)
        super().__init__(**layers)
        
    def __call__(self,x,ksize=3,stride=2,pad=1):
        h = self.conv1(x)
        h = F.relu(h)
        h = self.bnorm1(h)
        if self.pooling == 1:
            h = F.max_pooling_2d(h,ksize=ksize,stride=stride,pad=pad)

        return h
    
    
class PixelShuffler(Chain):
    def __init__(self,ch,r=2):
        self.r = r
        self.ch = ch
        super().__init__()
    
    def __call__(self,x):    
        [batchsize,ch,Ny,Nx] = x.shape
        ch_y = ch//(self.r**2)
        Ny_y = Ny*self.r
        Nx_y = Nx*self.r
        h = F.reshape(x, (batchsize, self.r, self.r, ch_y, Ny, Nx))
        h =  F.transpose(h, (0, 3, 4, 1, 5, 2))
        y = F.reshape(h, (batchsize, ch_y, Ny_y, Nx_y))
        return y

    
def temp_image(epoch,filename,xtest,ztest,gen,dis,Nfig = 3):
    print('epoch',epoch)
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):    
        ytest = gen(ztest)
        score_true = dis(xtest)
        score_false = dis(ytest)
    plt.figure(figsize=(12,12))
    for k in range(Nfig):
        plt.subplot(1,Nfig,k+1)
        plt.title("{}".format(score_true[k].data))
        plt.axis("off")
        plt.imshow(cuda.to_cpu(xtest[k,:,:,:]).transpose(1,2,0))
    plt.show()
    plt.figure(figsize=(12,12))
    for k in range(Nfig):
        plt.subplot(1,Nfig,k+1)
        plt.title("{}".format(score_false[k].data))
        plt.axis("off")
        plt.imshow(cuda.to_cpu(ytest[k,:,:,:].data).transpose(1,2,0))
    plt.savefig(filename+'_{0:03d}.png'.format(epoch))
    plt.show()
    
    

def temp_image2(epoch,filename,dataA,dataB,gen_AtoB,gen_BtoA,dis_A,dis_B):
    print('epoch',epoch)
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):   
        xtestAB = gen_AtoB(cuda.to_gpu(dataA))
        scoreAB = dis_B(xtestAB)
        xtestABA = gen_BtoA(xtestAB)
        xtestBA = gen_BtoA(cuda.to_gpu(dataB))
        scoreBA = dis_A(xtestBA)
        xtestBAB = gen_AtoB(xtestBA)
    kA = np.random.randint(len(dataA))
    kB = np.random.randint(len(dataB))
    plt.figure(figsize=(12,9))
    plt.subplot(3,2,1)
    plt.axis("off")
    plt.title("image A")
    plt.imshow(dataA[kA,:,:,:].transpose(1,2,0))
    plt.subplot(3,2,2)
    plt.axis("off")
    plt.imshow(dataB[kB,:,:,:].transpose(1,2,0))
    plt.axis("off")
    plt.title("image B")
    plt.subplot(3,2,3)
    plt.axis("off")
    plt.title("{}".format(cuda.to_cpu(scoreAB[kA].data)))
    plt.imshow(cuda.to_cpu(xtestAB[kA,:,:,:].data).transpose(1,2,0))
    plt.subplot(3,2,4)
    plt.axis("off")
    plt.title("{}".format(cuda.to_cpu(scoreBA[kB].data)))
    plt.imshow(cuda.to_cpu(xtestBA[kB,:,:,:].data).transpose(1,2,0))
    plt.subplot(3,2,5)
    plt.axis("off")
    plt.title("A to B to A")
    plt.imshow(cuda.to_cpu(xtestABA[kA,:,:,:].data).transpose(1,2,0))
    plt.subplot(3,2,6)
    plt.axis("off")
    plt.title("B to A to A")
    plt.imshow(cuda.to_cpu(xtestBAB[kB,:,:,:].data).transpose(1,2,0))
    plt.savefig(filename+'_{0:03d}.png'.format(epoch))
    plt.show()
    
def save_model(NN,filename):
    NN.to_cpu()
    ser.save_hdf5(filename+'.hd5', NN, compression=4)
    NN.to_gpu()

def load_model(NN,filename):
    ser.load_hdf5(filename+'.hd5', NN)
    NN.to_gpu()

def data_divide(Dtrain,D,xdata,tdata,shuffle='on'):
    if shuffle == 'on':
        index = np.random.permutation(range(D))
    elif shuffle == 'off':
        index = np.arange(D)
    else:
        print('error')
    xtrain = xdata[index[0:Dtrain],:]
    ttrain = tdata[index[0:Dtrain]]
    xtest = xdata[index[Dtrain:D],:]
    ttest = tdata[index[Dtrain:D]]
    return xtrain,xtest,ttrain,ttest

def plot_result(result,title,xlabel,ylabel,ymin=0.0,ymax=1.0):
    Tall = len(result)
    plt.figure(figsize=(8,6))
    plt.plot(range(Tall), result)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([0,Tall])
    plt.ylim([ymin,ymax])
    plt.show()        

def plot_result2(result1,result2,title,xlabel,ylabel,ymin=0.0,ymax=1.0):
    Tall = len(result1)
    plt.figure(figsize=(8,6))
    plt.plot(range(Tall), result1)
    plt.plot(range(Tall), result2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([0,Tall])
    plt.ylim([ymin,ymax])
    plt.show()        

def learning_regression(model,optNN,data,result,T=10): 
    for time in tqdm(range(T)):
        optNN.target.cleargrads()
        ytrain = model(data[0])
        loss_train = F.mean_squared_error(ytrain,data[2])
        loss_train.backward()
        optNN.update()

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        ytest = model(data[1])
    loss_test = F.mean_squared_error(ytest,data[3])
    result[0].append(cuda.to_cpu(loss_train.data))
    result[1].append(cuda.to_cpu(loss_test.data))

def learning_classification(model,optNN,data,result,T=10): 
    for time in range(T):
        optNN.target.cleargrads()
        ytrain = model(data[0])
        loss_train = F.softmax_cross_entropy(ytrain,data[2])
        acc_train = F.accuracy(ytrain,data[2])
        loss_train.backward()
        optNN.update()

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        ytest = model(data[1])
    loss_test = F.softmax_cross_entropy(ytest,data[3])
    acc_test = F.accuracy(ytest,data[3]) 
    result[0].append(cuda.to_cpu(loss_train.data))
    result[1].append(cuda.to_cpu(loss_test.data))
    result[2].append(cuda.to_cpu(acc_train.data))
    result[3].append(cuda.to_cpu(acc_test.data))


def learning_GAN_log(generator,discliminator,optgen,optdis,data,result,T = 200):
    for time in range(T):
        
        optgen.target.cleargrads()
        ytemp = generator(data[1])
        with chainer.using_config('train', False):
            ytrain_false = discliminator(ytemp)
        loss_train_gen = 0.5*F.mean(F.softplus(-ytrain_false))
        loss_train_gen.backward()
        optgen.update()
        
        #偽物画像
        optdis.target.cleargrads()
        ytrain_false = discliminator(ytemp)
        ytrain_true = discliminator(data[0])
        loss1 = 0.5*F.mean(F.softplus(ytrain_false))
        loss2 = 0.5*F.mean(F.softplus(-ytrain_true))
        loss_train_dis = loss1+loss2
        loss_train_dis.backward()
        optdis.update()       
          
    result[0].append(cuda.to_cpu(loss_train_gen.data))
    result[1].append(cuda.to_cpu(loss1.data))
    result[2].append(cuda.to_cpu(loss2.data))

def learning_GAN(generator,discliminator,optgen,optdis,data,result,T = 200):
    for time in range(T):
        
        optgen.target.cleargrads()
        ytemp = generator(data[1])
        with chainer.using_config('train', False):
            ytrain_false = discliminator(ytemp)
        loss_train_gen = 0.5*F.mean((ytrain_false-1.0)**2)
        loss_train_gen.backward()
        optgen.update()

        optdis.target.cleargrads()
        ytrain_false = discliminator(ytemp.data)
        ytrain_true = discliminator(data[0])
        loss1 = 0.5*F.mean((ytrain_false)**2,axis = (0,1))
        loss2 = 0.5*F.mean((ytrain_true-1.0)**2)
        loss_train_dis = loss1+loss2
        loss_train_dis.backward()
        optdis.update()       
          
    result[0].append(cuda.to_cpu(loss_train_gen.data))
    result[1].append(cuda.to_cpu(loss1.data))
    result[2].append(cuda.to_cpu(loss2.data))

        
def learning_consist(gen_BtoA,gen_AtoB,
                          optgen_BtoA,optgen_AtoB,data,T = 5):
    a = 10
    for time in range(T):
        optgen_BtoA.target.cleargrads()
        optgen_AtoB.target.cleargrads()
        ytemp1 = gen_BtoA(data[1])
        ytemp2 = gen_AtoB(data[0])
        loss_train = 0.5*a*F.mean_absolute_error(ytemp1,data[1])\
                     + 0.5*a*F.mean_absolute_error(ytemp2,data[0])
        loss_train.backward()
        result = loss_train.data
        optgen_BtoA.update()
        optgen_AtoB.update()
        
def learning_L1(gen_BtoA,gen_AtoB,
                     optgen_BtoA,optgen_AtoB,data,T = 5):
    a = 10
    for time in range(T):
        optgen_BtoA.target.cleargrads()
        optgen_AtoB.target.cleargrads()
        ytemp1 = gen_BtoA(data[0])
        ytrain1 = gen_AtoB(ytemp1)
        ytemp2 = gen_AtoB(data[1])
        ytrain2 = gen_BtoA(ytemp2)
        loss_train = 0.5*a*F.mean_absolute_error(ytrain1,data[0])\
                     + 0.5*a*F.mean_absolute_error(ytrain2,data[1])
        loss_train.backward()
        result = loss_train.data
        optgen_BtoA.update()
        optgen_AtoB.update()
              


def labeled64(labeled_data):
    data, label = labeled_data
    data = data.astype(np.uint8)
    data = im.fromarray(data.transpose(1, 2, 0))
    data = data.resize((64,64), im.BICUBIC)
    data = np.asarray(data).transpose(2, 0, 1)
    data = data.astype(np.float32)/255
    return data, label



def shift_labeled(labeled_data):
    data, label = labeled_data
    
    z_h = Ny*(2.0*np.random.rand(1)-1.0)*0.3
    z_v = Nx*(2.0*np.random.rand(1)-1.0)*0.3
    data = np.roll(data,int(z_h),axis=1)
    data = np.roll(data,int(z_v),axis=2)
    
    return data, label

def flip_labeled(labeled_data):
    data, label = labeled_data

    z = np.random.randint(2)
    if z == 1:
        data = data[:,::-1,:]

    z = np.random.randint(2)
    if z == 1:
        data = data[:,:,::-1]

    z = np.random.randint(2)
    if z == 1:
        data = data.transpose(0,2,1)
        
    return data, label

def add_labeled_data(folder, i, all_list):
    image_files = os.listdir(folder)
    for k in range(len(image_files)):
        labeled_file = (folder+"/"+image_files[k],i)
        all_list.append(labeled_file)
    return all_list

def check_network(x,Link):
    print("input:", x.shape)
    h = Link(x)
    print("output:", h.shape)
    return h