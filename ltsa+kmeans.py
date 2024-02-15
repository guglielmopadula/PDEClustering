from sklearn.manifold import LocallyLinearEmbedding
import numpy as np
from sklearn.cluster import KMeans
laplace=np.load("laplace.npy")
stokes=np.load("stokes.npy")
import matplotlib.pyplot as plt
np.random.shuffle(laplace)
np.random.shuffle(stokes)
laplace_train=laplace[:500]
laplace_test=laplace[500:]
stokes_train=stokes[:500]
stokes_test=stokes[500:]
data_train=np.concatenate((laplace_train,stokes_train))
data_test=np.concatenate((laplace_test,stokes_test))
embedding = LocallyLinearEmbedding(n_components=2,method='ltsa')
embedding.fit(data_train)
red_train=embedding.transform(data_train)
red_test=embedding.transform(data_test)
y_train=np.concatenate((np.zeros(500,dtype=np.int64),np.ones(500,dtype=np.int64)))
y_test=np.concatenate((np.zeros(100,dtype=np.int64),np.ones(100,dtype=np.int64)))

y_1=y_train.copy()
y_2=(y_train+1)%2

kmeans=KMeans(n_clusters=2)
kmeans.fit(red_train)
y_hat=kmeans.predict(red_train)

err_1=np.sum(np.abs(y_hat-y_1))/1000
err_2=np.sum(np.abs(y_hat-y_2))/1000




a=np.argmin([err_1,err_2])
print("Train missclassification rate: ",np.min([err_1,err_2]))
if a>0:
    y_train=(y_train+1)%2
    y_test=(y_test+1)%2


y_hat=kmeans.predict(red_test)
err_1=np.sum(np.abs(y_hat-y_test))/200
err_2=np.sum(np.abs(y_hat-(y_test+1)%2))/200
print("Test missclassification rate: ",np.min([err_1,err_2]))
