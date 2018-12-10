# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 12:48:22 2018

@author: lj
"""
import tensorflow as tf
import numpy as np

def load_data(file_name):
    '''导入数据
    input:  file_name(string):文件的存储位置
    output: feature_data(mat):特征
            label_data(mat):标签
            n_class(int):类别的个数
    '''
    # 1、获取特征
    f = open(file_name)  # 打开文件
    feature_data = []
    label_tmp = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label_tmp.append(int(lines[-1]))      
        feature_data.append(feature_tmp)
    f.close()  # 关闭文件
    
    # 2、获取标签
    m = len(label_tmp)
    n_class = len(set(label_tmp))  # 得到类别的个数
    
    label_data = np.mat(np.zeros((m, n_class)))
    for i in range(m):
        label_data[i, label_tmp[i]] = 1
    
    return np.mat(feature_data), label_data

class RBF_NN():
    def __init__(self,hidden_nodes,input_data_trainX,input_data_trainY):
        self.hidden_nodes = hidden_nodes #隐含层节点数
        self.input_data_trainX = input_data_trainX #训练样本的特征
        self.input_data_trainY = input_data_trainY #训练样本的标签
    
    def fit(self):
        '''模型训练
        '''
        # 1.声明输入输出的占位符
        n_input = (self.input_data_trainX).shape[1]
        n_output = (self.input_data_trainY).shape[1]
        X = tf.placeholder('float',[None,n_input],name = 'X')
        Y = tf.placeholder('float',[None,n_output],name = 'Y')
        # 2.参数设置
        ## RBF函数参数
        c = tf.Variable(tf.random_normal([self.hidden_nodes,n_input]),name = 'c')
        delta = tf.Variable(tf.random_normal([1,self.hidden_nodes]),name = 'delta')
        ## 隐含层到输出层权重和偏置
        W = tf.Variable(tf.random_normal([self.hidden_nodes,n_output]),name = 'W')
        b = tf.Variable(tf.random_normal([1,n_output]),name = 'b')
        # 3.构造前向传播计算图
        ## 隐含层输出
        ### 特征样本与RBF均值的距离
        dist = tf.reduce_sum(tf.square(tf.subtract(tf.tile(X,[self.hidden_nodes,1]),c)),1)  
        dist = tf.multiply(1.0,tf.transpose(dist))
        ### RBF方差的平方
        delta_2 = tf.square(delta)
        ### 隐含层输出
        RBF_OUT = tf.exp(tf.multiply(-1.0,tf.divide(dist,tf.multiply(2.0,delta_2))))           
        ## 输出层输入
        output_in = tf.matmul(RBF_OUT,W) + b
        ## 输出层输出
        y_pred = tf.nn.sigmoid(output_in)
        # 4.声明代价函数优化算法
        cost = tf.reduce_mean(tf.pow(Y - y_pred,2)) #损失函数为均方误差
        train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) #优化算法为梯度下降法
        
        # 5.反向传播求参数
        trX = self.input_data_trainX
        trY = self.input_data_trainY
        
        with tf.Session() as sess:
            ##初始化所有参数
            tf.global_variables_initializer().run()
            for epoch in range(100):
                for i in range(len(trX)):
                    feed = {X:trX[i],Y:trY[i]}
                    sess.run(train_op,feed_dict = feed)
                if epoch % 10.0 == 0:
                    total_loss = 0.0
                    for j in range(len(trX)):
                        total_loss += sess.run(cost,feed_dict = {X:trX[j],Y:trY[j]})
                    print('Loss function at step %d is %s'%(epoch,total_loss / len(trX)))
            print('Training complete!')

            W = W.eval()
            b = b.eval()
            c = c.eval()
            delta = delta.eval()
            pred_trX = np.mat(np.zeros((len(trX),n_output)))
            ## 训练准确率
            correct_tr = 0.0
            for i in range(len(trX)):
                pred_tr = sess.run(y_pred,feed_dict = {X:trX[i]})
                pred_trX[i,:] = pred_tr
                if np.argmax(pred_tr,1) == np.argmax(trY[i],1):
                    correct_tr += 1.0
            print('Accuracy on train set is :%s'%(correct_tr/len(trX)))
            self.save_model('RBF_predict_results.txt',pred_trX)
            
    def save_model(self,file_name,weights):
        '''保存模型(保存权重weights)
        input：file_name(string):文件名
               weights(mat)：权重矩阵
        '''
        f_w = open(file_name,'w')
        m,n = np.shape(weights)
        for i in range(m):
            w_tmp = []
            for j in range(n):
                w_tmp.append(str(weights[i,j]))
            f_w.write('\t'.join(w_tmp)+'\n')
        f_w.close()
            
if __name__ == '__main__':
    print('------------------------1.Load Data---------------------')
    trainX, trainY = load_data('data.txt')
    print('------------------------2.parameter setting-------------')
    hidden_nodes = 20
    input_data_trainX = trainX
    input_data_trainY = trainY
    rbf = RBF_NN(hidden_nodes,input_data_trainX,input_data_trainY)
    rbf.fit()
