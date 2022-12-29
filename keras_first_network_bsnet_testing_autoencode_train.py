# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import tensorflow as keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import Model
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing

from datetime import datetime
import time

@tf.function
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost
    
# load the dataset
#dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
#X = np.concatenate((dataset[:,0:8],-dataset[:,0:8],np.ones((dataset.shape[0],1))),axis=1)
#X=dataset[:,0:8]
#y = dataset[:,8]

#scaler = preprocessing.StandardScaler().fit(X)
#X = scaler.transform(X)

# Split the remaining data to train and validation
#x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=True,random_state=1000)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#x_train=np.reshape(x_train,(-1,28*28))
#x_test=np.reshape(x_test,(-1,28*28))

print(np.shape(y_train))
print(np.max(x_train))

temp_x_train=x_train.copy()
temp_x_train[temp_x_train>0.5]=1
temp_x_train[temp_x_train<=0.5]=0
temp_x_test=x_test.copy()
temp_x_test[temp_x_test>0.5]=1
temp_x_test[temp_x_test<=0.5]=0
#y_train=np.concatenate((x_train,1-x_train),axis=1)
#y_test=np.concatenate((x_test,1-x_test),axis=1)

print(np.shape(y_train))



#plt.imshow(x_train[0,:,:])
#plt.show()
#exit(1)

x_train=x_train.reshape((-1, 28*28))
x_test=x_test.reshape((-1, 28*28))


dim=8#8
'''
x_train_new=np.zeros((x_train.shape[0],x_train.shape[1],dim))
a1=x_train*dim-(x_train*dim).astype(int)
a2=(x_train*dim).astype(int)
for i in range(0,dim):
    temp=a1.copy()
    temp[a2!=i]==0
    x_train_new[:,:,i]=temp
x_train=np.reshape(x_train_new,(-1,28*28*dim))
print('convert each input into 8 dimensions')
x_test_new=np.zeros((x_test.shape[0],x_test.shape[1],dim))
a1=x_test*dim-(x_test*dim).astype(int)
a2=(x_test*dim).astype(int)
for i in range(0,dim):
    temp=a1.copy()
    temp[a2!=i]==0
    x_test_new[:,:,i]=temp
x_test=np.reshape(x_test_new,(-1,28*28*dim))
print('convert each input into 8 dimensions')
#'''

y_test_save=y_test.copy()
y_train=x_train.copy()
y_test=x_test.copy()

np.random.seed(1000)
a=(np.array(np.random.rand(x_train.shape[0],x_train.shape[1])>0.3)).astype(float)#0.3#,x_train.shape[2]
#a=1*(np.array(np.random.rand(x_train.shape[0],x_train.shape[1],x_train.shape[2]))).astype(float)#0.3
#np.random.seed(1010)
#b=1*(np.array(np.random.rand(x_train.shape[0],x_train.shape[1],x_train.shape[2]))).astype(float)#0.3
#np.random.seed(1020)
#c=1*(np.array(np.random.rand(x_train.shape[0],x_train.shape[1],x_train.shape[2]))).astype(float)#0.3
x_train=np.maximum(a,x_train)
#x_train=a+x_train

#'''
x_train_new=np.zeros((x_train.shape[0],x_train.shape[1],dim))
a2=(x_train*((2**dim)*1-1e-10)).astype(int)
for i in range(0,dim):
    x_train_new[:,:,i]=a2%2
    a2=a2//2
    #x_train_new[:,:,i]=(a2%(2*256))/(2*256)
    #a2=a2//2
x_train=np.reshape(x_train_new,(-1,28*28*dim))
print('convert each input into 8 dimensions')
x_test_new=np.zeros((x_test.shape[0],x_test.shape[1],dim))
a2=(x_test*((2**dim)*1-1e-10)).astype(int)
for i in range(0,dim):
    x_test_new[:,:,i]=a2%2
    a2=a2//2
    #x_test_new[:,:,i]=(a2%(2*256))/(2*256)
    #a2=a2//2
x_test=np.reshape(x_test_new,(-1,28*28*dim))
print('convert each input into 8 dimensions')
#'''

#y_train=x_train.copy()
#y_test=x_test.copy()

'''
#test PCA
pca = PCA(n_components=100)
#pca.fit(x_train)

x_train2=pca.fit_transform(x_train)
print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))
print(np.shape(x_train2))
x_train3=pca.inverse_transform(x_train2)

temp2=x_train[1000,:]
#temp2=np.reshape(temp2,(28,28))
temp2=np.reshape(temp2,(28,28,32))
ssum=0
for i in range(0,temp2.shape[2]):
    ssum=ssum+temp2[:,:,i]*i
#temp2=temp2[:,:,0]*1+temp2[:,:,1]*2+temp2[:,:,2]*4+temp2[:,:,3]*8+temp2[:,:,4]*16+temp2[:,:,5]*32+temp2[:,:,6]*64+temp2[:,:,7]*128
#temp2=temp2/256
temp2=ssum

temp=x_train3[1000,:]
#temp=np.reshape(temp,(28,28))
temp=np.reshape(temp,(28,28,32))
ssum=0
for i in range(0,temp.shape[2]):
    ssum=ssum+temp[:,:,i]*i
#temp=temp[:,:,0]*1+temp[:,:,1]*2+temp[:,:,2]*4+temp[:,:,3]*8+temp[:,:,4]*16+temp[:,:,5]*32+temp[:,:,6]*64+temp[:,:,7]*128
#temp=temp/256
temp=ssum
plt.figure()
plt.imshow(temp2)
plt.figure()
plt.imshow(temp)
plt.show()
'''

'''
# define the keras model
model = Sequential()
model.add(Dense(96, input_dim=8, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

'''
#'''#tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=100.0, rate=1.0, axis=0)
#bias_constraint=tf.keras.constraints.NonNeg()
x_in=Input(shape=(28*28*dim,))#
#x_in = Input(shape=(8,))

initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=0.05, seed=None)

#"""
x = Dense(100, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_in, -x_in]))
x_1 = Dense(100, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x, -x]))#400
x_2 = Dense(100, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_1, -x_1,x,-x]))
#x_2_1=Dense(800, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x, -x]))
x_3 = Dense(100, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_2, -x_2,x,-x]))
#x_3=x_2_1+x_3_1
x_4 = Dense(100, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_3, -x_3,x,-x]))
#x_4_1=Dense(800, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x, -x]))
x_5 = Dense(100, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_4, -x_4]))#([x_4, -x_4]))#,x_in,-x_in
#x_5=x_5_1+x_4_1

x_6 = Dense(100, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_5, -x_5]))
x_7 = Dense(100, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_6, -x_6,x_5, -x_5]))
#x_7=Dense(800, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg(),)(Concatenate(axis=1)([x_5, -x_5]))
x_8 = Dense(100, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_7, -x_7,x_5, -x_5]))
#x_8=x_8_1+x_7_1
x_9 = Dense(100, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_8, -x_8,x_5, -x_5]))
#x_9_1 = Dense(800, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_5, -x_5]))
x_10 = Dense(100, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_9, -x_9,x_5, -x_5]))#([x_9, -x_9]))
#x_10=x_10_1+x_9_1
x_out = Dense(28*28*dim, activation='sigmoid', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_10, -x_10]))#,x_5,-x_5

x_out2=tf.reshape(x_out,[-1,28,28,dim])

#for i in range(0,32):
#    x_out3=x_out2[:,:,:,i]*(2**i)
#x_out3=x_out3/(2**32)
x_out3=(x_out2[:,:,:,0]*1+x_out2[:,:,:,1]*2+x_out2[:,:,:,2]*4+x_out2[:,:,:,3]*8+x_out2[:,:,:,4]*16+x_out2[:,:,:,5]*32+x_out2[:,:,:,6]*64+x_out2[:,:,:,7]*128)/256
x_out4=tf.reshape(x_out3,[-1,28*28])
#"""

"""
x = Dense(100, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_in, -x_in]))
x_1 = Dense(50, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x, -x]))#400
x_2 = Dense(30, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_1, -x_1]))
x_3 = Dense(10, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_2, -x_2]))
x_4 = Dense(10, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_3, -x_3]))
x_5 = Dense(10, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_4, -x_4]))

x_6 = Dense(10, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_5, -x_5]))
x_7 = Dense(10, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_6, -x_6]))
x_8 = Dense(30, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_7, -x_7]))
x_9 = Dense(50, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_8, -x_8]))
x_10 = Dense(100, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_9, -x_9]))
x_out = Dense(28*28, activation='relu', kernel_constraint=tf.keras.constraints.NonNeg())(Concatenate(axis=1)([x_10, -x_10]))
"""

#'''



model = Model(inputs=x_in, outputs=x_out4)

# compile the keras model
optimizer = keras.optimizers.Adam(lr=0.0001)#lr=0.001#0.0001
#model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])#optimizer='adam'
##model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizer, metrics=['accuracy'])
###model.compile(loss=macro_soft_f1, optimizer=optimizer, metrics=['accuracy'])
model.compile(loss='mae', optimizer=optimizer)
# fit the keras model on the dataset
###model.fit(x=x_train, y=y_train, epochs=150, batch_size=100, validation_data=(x_test, y_test))#1500 #epochs=150, batch_size=100#epochs=1500, batch_size=60000

'''
# Restore the weights
model.load_weights('./checkpoints/my_checkpoint13')

print(model.summary())
print(len(model.layers))

newmodel = Model(inputs=model.get_layer('dense_5').input, outputs=model.layers[-1].output)#dense_6#.get_layer("dense_11")
newmodel2 = Model(inputs=model.layers[0].input, outputs=model.get_layer('dense_4').output)#dense_5

"""
test_mid=newmodel2.predict(np.reshape(x_test,(-1,28*28*8)))
pca = PCA(n_components=2)
test_mid2=pca.fit_transform(test_mid)
#scaler = preprocessing.StandardScaler().fit(test_mid)
#test_mid2 = scaler.transform(test_mid)
for i in range(0,10):
    print(test_mid2[y_test_save==i,:])
    plt.figure()
    plt.imshow(test_mid[y_test_save==i,:])
plt.show()    
for i in range(0,10):
    labels=y_test_save.copy()
    labels=labels.astype(int)
    labels[labels!=i]=-1
    labels[labels==i]=1
    labels[labels==-1]=0
    plt.scatter(test_mid2[labels==0,0],test_mid2[labels==0,1])
    plt.scatter(test_mid2[labels==1,0],test_mid2[labels==1,1])
    plt.show()
    #lda = LinearDiscriminantAnalysis(n_components=1)
    #test_mid3 = lda.fit(test_mid2, labels).transform(test_mid2)
    #plt.hist(test_mid3[labels==0])
    #plt.hist(test_mid3[labels==1])
    #plt.show()
    #plt.scatter(test_mid2[y_test_save==i,0],test_mid2[y_test_save==i,1])
plt.show()
#"""

test_out_big=np.zeros((28*10,28*10))
test_in_big=np.zeros((28*10,28*10))
test_in_save_big=np.zeros((28*10,28*10))
np.random.seed(7777)
for i in range(0,10):
    for j in range(0,10):
        b=int(np.floor(np.random.rand(1)*10000))
        test_in=np.reshape(x_test[b,:],(28,28,8))
        #np.random.seed(1000)
        a=(np.array(np.random.rand(28,28)>0.3)).astype(float)#0.3,0.95
        a=np.tile(np.reshape(a,(28,28,1)),(1,1,8))
        #print(np.shape(a))
        test_in_save=test_in.copy()
        test_in=np.maximum(a,test_in)
        test_out=model.predict(np.reshape(test_in,(1,28*28*8)))
        #print(test_out)
        test_out=np.reshape(test_out,(28,28))
        #test_out=test_out[1,:,:]
        test_out_big[i*28:(i+1)*28,j*28:(j+1)*28]=np.minimum(test_out,1) #test_out[:,:,0]*1+test_out[:,:,1]*2+test_out[:,:,2]*4+test_out[:,:,3]*8+test_out[:,:,4]*16+test_out[:,:,5]*32+test_out[:,:,6]*64+test_out[:,:,7]*128
        test_in_big[i*28:(i+1)*28,j*28:(j+1)*28]=test_in[:,:,0]*1+test_in[:,:,1]*2+test_in[:,:,2]*4+test_in[:,:,3]*8+test_in[:,:,4]*16+test_in[:,:,5]*32+test_in[:,:,6]*64+test_in[:,:,7]*128
        #test_in_big[i*28:(i+1)*28,j*28:(j+1)*28]=test_in[:,:,0]*1+test_in[:,:,1]*2+test_in[:,:,2]*4+test_in[:,:,3]*8
        test_in_save_big[i*28:(i+1)*28,j*28:(j+1)*28]=test_in_save[:,:,0]*1+test_in_save[:,:,1]*2+test_in_save[:,:,2]*4+test_in_save[:,:,3]*8+test_in_save[:,:,4]*16+test_in_save[:,:,5]*32+test_in_save[:,:,6]*64+test_in_save[:,:,7]*128
plt.figure()
plt.imshow(test_in_save_big)
plt.figure()
plt.imshow(test_in_big)
plt.figure()
plt.imshow(test_out_big)
plt.show()


np.random.seed(int(time.time()))
n=2
index=np.where(y_test_save==2)
index=index[0]
b=(np.floor(np.random.rand(n)*10000)).astype(int)
#b=(np.floor(np.random.rand(n)*len(index))).astype(int)
#print(index)
#b=index[b]
#b2=(np.random.rand(n)*10000)
#b2=b2/n
test_out_big=np.zeros((28*1,28*30))
for i in range(0,30):
    b2=np.array([i/30,1-i/30])
    #b2=(b2-0.5)*2+0.5
    print(b2)
    #b3=0.5#(np.random.rand(1)
    test_in=np.reshape(x_test[b,:],(n,28,28,8))
    #test_in2=np.reshape(x_test[b2,:],(28,28,4))
    test_mid=newmodel2.predict(np.reshape(x_test[b,:],(n,28*28*8)))#n
    #print(np.shape(test_mid))
    #test_mid2=newmodel2.predict(np.reshape(test_in2,(1,28*28*4)))
    temp=np.reshape(np.sum(test_mid*np.tile(np.reshape(b2,(n,1)),(1,100)),axis=0),(1,100))
    print(np.shape(temp))
    #temp=test_mid*b3+test_mid2*(1-b3)
    #test_mid=np.squeeze(test_mid)
    #temp=np.concatenate((test_mid,-test_mid,np.mean(x_test[b,:],axis=0),-np.mean(x_test[b,:],axis=0)),axis=0)#axis=1
    temp=np.concatenate((temp,-temp),axis=1)#axis=1
    #temp=np.reshape(temp,(1,len(temp)))
    test_out=newmodel.predict(temp)
    test_out=np.reshape(test_out,(28,28))
    j=0
    test_out_big[j*28:(j+1)*28,i*28:(i+1)*28]=test_out
    #print(b3)
    #for i in range(0,n):
        #plt.figure()
        #plt.imshow(test_in[:,:,0]*1+test_in[:,:,1]*2+test_in[:,:,2]*4+test_in[:,:,3]*8)
        #plt.imshow(test_in[i,:,:,0]*1+test_in[i,:,:,1]*2+test_in[i,:,:,2]*4+test_in[i,:,:,3]*8+test_in[i,:,:,4]*16+test_in[i,:,:,5]*32+test_in[i,:,:,6]*64+test_in[i,:,:,7]*128)
    #plt.figure()
    #plt.imshow(test_in2[:,:,0]*1+test_in2[:,:,1]*2+test_in2[:,:,2]*4+test_in2[:,:,3]*8)
    #plt.figure()
    #plt.imshow(test_out)
    #plt.show()
    #exit(1)
plt.figure()
plt.imshow(test_out_big)
plt.show()


test_out_big=np.zeros((28*10,28*10))
for i in range(0,10):
    for j in range(0,10):
        temp=np.random.rand(1,10*10)#*100-50
        #temp=np.concatenate((temp,-temp),axis=1)
        test_out = newmodel.predict(temp)#np.ones((1,10)))
        test_out=np.reshape(test_out,(28,28))
        test_out_big[i*28:(i+1)*28,j*28:(j+1)*28]=test_out
plt.figure()
plt.imshow(test_out_big)
plt.show()
#exit(1)

# evaluate model on test set
#mae = model.evaluate(x_test, y_test, verbose=0)
# store result
#print('>%.3f' % mae)


exit(1)
#'''

# fit model
model.fit(x_train, y_train, verbose=1, epochs=150,batch_size=100, validation_data=(x_test, y_test))#150,100
# evaluate model on test set
mae = model.evaluate(x_test, y_test, verbose=0)
# store result
print('>%.3f' % mae)
        
# evaluate the keras model
#_, accuracy = model.evaluate(X, y)
##_, accuracy = model.evaluate(x_test, y_test)
##print('Accuracy: %.2f' % (accuracy*100))

# Save the weights
model.save_weights('./checkpoints/my_checkpoint13')



