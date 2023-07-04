# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt
import logging
import time


FORMAT = '%(asctime)s %(clientip)-15s %(user)-8s %(message)s'
logging.basicConfig(format=FORMAT)

#load data
arc = np.load('mnist.npz')

x_train = arc['arr_0']
y_train = arc['arr_1']
x_test  = arc['arr_2']
y_test  = arc['arr_3']

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)



# Show image number 15, and write in the title what digit it should correspond to
N=15
plt.imshow(x_train[N], cmap='gray_r')
plt.title('Hand written digit '+str(y_train[N]))


"""
Assignment 1
"""

#put this inside a class
#see if any construcotr __init__ can be uses
#saves the weights and bisases

def d_infty(a,b):
    #computes norm in L_inf
    
    logging.info("computing d_infty")
    
    return np.max(np.absolute((a - b)))

def d_one(a,b):
    #computes norm in L_1

    logging.info("computing d_one")

    
    return np.sum(np.absolute((a - b)))

def d_two(a,b):
    #computes norm in L_2
    
    logging.info("computing d_two")
    return np.sqrt(np.sum(np.square((a - b))))





"""
Assignment 2
"""

def dist(x_train, N, norm = d_one):
    #computes distances between two objects of x_train

    logging.info("computing dist")

    
    #todo : write smater
    #if concat possible, use it
    #divide the data in several processors
    #then copy in lower half
    #time it

    D = np.zeros((N,N))
    #D = d_one(x_train[i, :, :], x_train[j, :, :])
    for i in range(N):
        #upper traingular matrix
        for j in range(i,N):
            D[i,j] = norm(x_train[i, :, :], x_train[j, :, :])
            #using the fact, the martrix is symmteric
            D[j,i] = D[i,j]
    
    logging.info("checking D == D^T")
    assert(np.all(D == np.transpose(D)))
    
    
    return D




"""
Assignment 3
"""


# plot the dist matrix D in infrinity, one and two norm for first 100 images.

N=100
D_inf = dist(x_train, N, norm=d_infty)
plt.imshow(D_inf, cmap='gray_r')
plt.title('Dist in Norm : d_infty for N = '+np.str(N))
plt.savefig("d_inf.png", dpi = 300)
plt.show()

D_1 = dist(x_train, N, norm=d_one)
plt.imshow(D_1, cmap='gray_r')
plt.title('Dist in Norm : d_one for N = '+np.str(N))
plt.savefig("d_one.png", dpi = 300)
plt.show()


D_2 = dist(x_train, N, norm=d_two)
plt.imshow(D_2, cmap='gray_r')
plt.title('Dist in Norm : d_two for N = '+np.str(N))
plt.savefig("d_two.png", dpi = 300)
plt.show()





"""
Assignment 4

Leove One out

"""



def loo(dist, N, norm = d_one):
    #Leave one out -> to compute effiency 

    logging.info("Implementing Leave one out")

    D1 = dist(x_train, N, norm=norm)

    error_counter = 0
    N1=0
    for i in range(N):
        #for j in range(N):
    
            #find the smallest j, other othan diag.
            D1[i,i] = 1e8
    
            j1 = np.argmin(D1[i,:])
            #j1 = np.where(np.concatenate(j1) !=j )
            #j1 = np.concatenate(j1)
            #for k in j1:
                #         N1+=1
            
            if (y_train[i] != y_train[j1]):
                    error_counter +=1
                    
    return error_counter / (N)





"""
Assignment 5

plot errors

"""

err = np.zeros((5, 4))
tim = np.zeros((5, 4))
i = 0
N1 = [100, 200, 400, 800, 1600]
for N in N1:
    start_time = time.time()
    err[i,0] =  loo(dist, N, norm = d_infty)
    current_time = time.time() - start_time
    tim[i,0] = current_time
    
    start_time = time.time()
    err[i,1] =  loo(dist, N, norm = d_one)
    current_time = time.time() - start_time
    tim[i,1] = current_time

    start_time = time.time()
    err[i,2] =  loo(dist, N, norm = d_two)
    current_time = time.time() - start_time
    tim[i,2] = current_time



    i+=1


# compute error plat


plt.loglog(N1,err[:,0], "r--", label = "d_inf")
plt.loglog(N1, err[:,1], "b--", label = "d_one")
plt.loglog(N1,err[:,2], "k--", label = "d_two")
plt.title("errors in different norms")
plt.xlabel("N")
plt.ylabel("Error")
plt.legend()
plt.show()


# compute time plat

plt.loglog(N1,tim[:,0], "r--", label = "d_inf")
plt.loglog(N1, tim[:,1], "b--", label = "d_one")
plt.loglog(N1,tim[:,2], "k--", label = "d_two")
plt.title("compute time in different norms and N")
plt.xlabel("N")
plt.ylabel("compute time")
plt.legend()
plt.show()


    
    



"""

Assigment 6

"""

def d_h1(a,b):
    
    # compute H1 norm

    
    logging.info("computing H1 norm")

    a, b = a.reshape(28, 28), b.reshape(28,28)
    #normaise
    a = a / np.sum(a)
    b = b / np.sum(b)
    
    x, y = np.arange(0, 28), np.arange(0, 28)
    
    #compute differnce
    a_b = a - b
    #compute gard in x directions
    grad_ab_x =  (a_b[1:, 1:] - a_b[:-1,1:])/(x[1:] - x[:-1]) 
    #compute grad in y direction
    grad_ab_y = (a_b[1:, 1:] - a_b[1:,:-1])/(y[1:] - y[:-1])
    
    #the require parts for the H1 norm, check return too
    d_h = np.square(grad_ab_x) + np.square(grad_ab_y) + np.square(a_b[1:, 1:])
    #d_h = np.square(grad_ab_x + grad_ab_y) + np.square(a_b[-1:, :-1])

    
    return np.sqrt( np.sum(d_h))


i = 0
N1 = [100, 200, 400, 800, 1600]
for N in N1:
    start_time = time.time()
    err[i,3] =  loo(dist, N, norm = d_h1)
    current_time = time.time() - start_time
    tim[i,3] = current_time
    i+=1
    


# plot error

plt.loglog(N1,err[:,0], "r--", label = "d_inf")
plt.loglog(N1, err[:,1], "b--", label = "d_one")
plt.loglog(N1,err[:,2], "k--", label = "d_two")
plt.loglog(N1,err[:,3], "g--", label = "d_H1")

plt.title("errors in different norms")
plt.xlabel("N")
plt.ylabel("Error")
plt.legend()
plt.savefig("error_for_diff_norms.png", dpi = 300)
plt.show()



#compute time plot


plt.loglog(N1,tim[:,0], "r--", label = "d_inf")
plt.loglog(N1, tim[:,1], "b--", label = "d_one")
plt.loglog(N1,tim[:,2], "k--", label = "d_two")
plt.loglog(N1,tim[:,3], "g--", label = "d_H1")

plt.title("compute time in different norms and N")
plt.xlabel("N")
plt.ylabel("compute time")
plt.legend()
plt.savefig("compute_time.png", dpi = 300)
plt.show()



"""

Assinment 7 : Balltree

"""


from sklearn.neighbors import BallTree

N2 = [200, 400, 800, 1600, 3200, 6400]



def ball_tree(N, k1, norm):
    
    #Ball Tree
    
    logging.info("Ball Tree")


    #build data strcuture
    X = x_train[:N,:,:].reshape((N, 28*28))
    tree = BallTree(X, metric=norm)

    
    # compute efficeny
    x_t = x_test[:,:,:].reshape((-1, 28*28))

    error_counter = 0
    for i in range(x_test.shape[0]):

        distt, k = tree.query(x_t[i, :].reshape(1,-1), k=k1)
        
        k = int(k)
        
        if y_train[k] != y_test[i]:
            error_counter+=1
            
    return error_counter / len(x_test)


err_2 = np.zeros((6, 4))
tim_2 = np.zeros((6, 4))
i=0
#case k = 1
k = 1
for N in N2:
    start_time = time.time()
    err_2[i, 0] = ball_tree(N, k, norm = d_infty)
    current_time = time.time() - start_time
    tim_2[i,0] = current_time
    
    start_time = time.time()
    err_2[i, 1] = ball_tree(N, k, norm = d_one)
    current_time = time.time() - start_time
    tim_2[i,1] = current_time
    
    start_time = time.time()
    err_2[i, 2] = ball_tree(N, k, norm = d_two)
    current_time = time.time() - start_time
    tim_2[i,2] = current_time
    
    start_time = time.time()
    err_2[i, 3] = ball_tree(N, k, norm = d_h1)
    current_time = time.time() - start_time
    tim_2[i,3] = current_time
    
    i+=1



#plot error for k = 1

plt.loglog(N2,err_2[:,0], "r--", label = "d_inf")
plt.loglog(N2, err_2[:,1], "b--", label = "d_one")
plt.loglog(N2,err_2[:,2], "k--", label = "d_two")
plt.loglog(N2,err_2[:,3], "g--", label = "d_H1")


plt.title("Balltree : Errors with diffnerent N and norms for k = " +np.str(k))
plt.xlabel("N")
plt.ylabel("Error")
plt.legend()
plt.savefig("ball_error_k="+np.str(k)+".png", dpi = 300)
plt.show()


#compute time plot


plt.loglog(N2,tim_2[:,0], "r--", label = "d_inf")
plt.loglog(N2, tim_2[:,1], "b--", label = "d_one")
plt.loglog(N2,tim_2[:,2], "k--", label = "d_two")
plt.loglog(N2,tim_2[:,3], "g--", label = "d_H1")

plt.title("Ball tree : compute time in different norms and N for k = " +np.str(k))
plt.xlabel("N")
plt.ylabel("compute time")
plt.legend()
plt.savefig("ball_compute_time_k="+np.str(k)+".png", dpi = 300)
plt.show()






#Ball tree with  k = 10



i=0
k = 10
err_3 = np.zeros((6, 4))
tim_3 = np.zeros((6, 4))

for N in N2:
    
    start_time = time.time()
    err_3[i, 0] = ball_tree(N, k, norm = d_infty)
    current_time = time.time() - start_time
    tim_3[i,0] = current_time
    
    start_time = time.time()
    err_3[i, 1] = ball_tree(N, k, norm = d_one)
    current_time = time.time() - start_time
    tim_3[i,1] = current_time

    start_time = time.time()
    err_3[i, 2] = ball_tree(N, k, norm = d_two)
    current_time = time.time() - start_time
    tim_3[i,2] = current_time
    
    start_time = time.time()
    err_3[i, 3] = ball_tree(N, k, norm = d_h1)
    current_time = time.time() - start_time
    tim_3[i,3] = current_time
    
    i+=1



#plot error for k = 10


plt.loglog(N2,err_2[:,0], "r--", label = "d_inf")
plt.loglog(N2, err_2[:,1], "b--", label = "d_one")
plt.loglog(N2,err_2[:,2], "k--", label = "d_two")
plt.loglog(N2,err_2[:,3], "g--", label = "d_H1")


plt.title("Balltree : Errors with diffnerent N and norms for k = " +np.str(k))
plt.xlabel("N")
plt.ylabel("Error")
plt.legend()
plt.savefig("ball_error_k="+np.str(k)+".png", dpi = 300)
plt.show()


#compute time plot


plt.loglog(N2,tim_3[:,0], "r--", label = "d_inf")
plt.loglog(N2, tim_3[:,1], "b--", label = "d_one")
plt.loglog(N2,tim_3[:,2], "k--", label = "d_two")
plt.loglog(N2,tim_3[:,3], "g--", label = "d_H1")

plt.title("Ball tree : compute time in different norms and N for k = " +np.str(k))
plt.xlabel("N")
plt.ylabel("compute time")
plt.legend()
plt.savefig("ball_compute_time_k="+np.str(k)+".png", dpi = 300)
plt.show()

    
    
