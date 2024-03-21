import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def soft_threshold(x,t):
    pos = np.maximum(x - t, 0)
    neg = np.minimum(x + t, 0)
    return pos+neg
   

# %%  ########### PROXIMAL GRADIENT DESCENT ###########
# Reference correct solution:
A = np.array([[1,2],[3,4]])
y = np.array([-2,3])
f = lambda x: 0.5*((y-A@x)**2).sum() + lam*np.abs(x).sum()
x0 = np.random.randn(2)
print(minimize(f,x0))


# Proximal GD
A = np.array([[1,2],[3,4]])
y = np.array([-2,3])
lam = 0.5
t = 0.01
iterations = 2000
x_new = np.zeros(2)
x = [x_new]
f = lambda x: 0.5*((y-A@x)**2).sum() + lam*np.abs(x).sum()
L = [f(x_new)]

for i in range(iterations-1):
    x_current = x[-1]
    grad_g = -A.T@(y-A@x_current)
    x_new_gd = x_current - t * grad_g; # x_k update before soft thresholding
    # apply soft thresholding:.
    x_new = soft_threshold(x_new_gd,lam*t)
    x.append(x_new)
    L.append(f(x_new))

plt.plot(L)
plt.title('Proximal GD')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.show()
# Accelerated Proximal Gradient Descent
A = np.array([[1,2],[3,4]])
y = np.array([-2,3])
lam = 0.5
t = 0.01
iterations = 2000
x_new = np.zeros(2)
x = [x_new,x_new]
f = lambda x: 0.5*((y-A@x)**2).sum() + lam*np.abs(x).sum()
L = [f(x_new)]
for i in range(iterations):
    x_current = x[-1]
    x_prev = x[-2]
    v = x_current + ((i-2)/(i+1))*(x_current-x_prev)
    grad_g = -A.T@(y - A@v)
    x_new_gd = v - t * grad_g
    x_new = soft_threshold(x_new_gd,lam*t)
    x.append(x_new)
    L.append(f(x_new))


plt.plot(L)
plt.title('Accelerated Proximal GD')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.show()

# %% ############ AUGMENTED LAGRANGIAN METHOD ############

# Augmented Lagrangian Method 
iterations = 100
u = 0
rho = 0.1
x_2 = 0 # x_2 remains fixed as we derived.
xs = []
L = []
for i in range(iterations):
    x_1 = (rho-u)/(1+rho)
    xs.append(x_1)
    u += rho*(x_1-1)
    L.append(0.5*x_1**2  + 0.5*x_2**2)

plt.plot(xs,label='$x_1$')
plt.title('Augmented Lagragian Method')
plt.plot(L,label='Objective')
plt.xlabel('Iteration')
plt.legend()
plt.show()

#  ADMM 
L = []
iterations = 100;
rho = 0.1;
w = 0; # intial value
x_2 = 0; # intial value.
x1s = []
x2s = []
for i in range(iterations):
    x_1 = -rho*(x_2-1+w)/(1+rho)
    x_2 = -rho*(x_1-1+w)/(1+rho)
    w = w + x_1 + x_2 -1
    L.append(0.5*x_1**2  + 0.5*x_2**2)
    x1s.append(x_1)
    x2s.append(x_2)
plt.plot(L,label='Objective')
plt.plot(x1s,label='$x_1$')
plt.plot(x2s,label='$x_2$')
plt.title('ADNM')
plt.xlabel('Iteration')
plt.legend()
plt.show()




# %% ############### COORDINATE DESCENT ALGORITHM ###############
A = np.array([[1,2],[3,4]]) # Is also gradient
y = np.array([-2,3])
f = lambda x: 0.5*((y-A@x)**2).sum() + lam*np.abs(x).sum()
lam = 0.5
x = np.zeros(2)
L = [f(x)]
x1s = [x[0]]
x2s = [x[1]]
for k in range(500):
    gamma1 = lam/(A[:,0]@A[:,0])
    gamma2 = lam/(A[:,1]@A[:,1])
    x1_ = A[:,0]@(y-A[:,1]*x[1])/(A[:,0]@A[:,0])
    x[0] = soft_threshold(x1_,gamma1)
    x2_ = A[:,1]@(y-A[:,0]*x[0])/(A[:,1]@A[:,1])
    x[1] = soft_threshold(x2_, gamma2)
    L.append(f(x))
    x1s.append(x1_)
    x2s.append(x2_)


plt.plot(L,label='Objective')
plt.plot(x1s,label='$x_1$')
plt.plot(x2s,label='$x_2$')
plt.title('ADNM')
plt.xlabel('Iteration')
plt.legend()
plt.show()


