# Part1 Task1
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def my_step(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def my_ramp(t):
    y = np.zeros(t.shape)
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y

# f1(t) = u(t-2) - u(t-9)
def f_1(t):
    y = my_step(t-2) - my_step(t-9)
    return y

steps = 1e-2
t = np.arange(0, 20+steps, steps)

plt.figure(figsize=(10,7))
plt.subplot(3, 1, 1)
plt.plot(t,f_1(t))
plt.title('Three User Defined Functions')
plt.ylabel('f1(t)')
plt.ylim([0,1.2])
plt.grid()
plt.show()

# f2(t) = e^(-t)*u(t)
def f_2(t):
    y = np.exp(-t)*my_step(t)
    return y

plt.subplot(3, 1, 2)
plt.plot(t,f_2(t))
plt.ylabel('f2(t)')
plt.ylim([0,1.2])
plt.grid()
plt.show()

# f3(t) = r(t-2)[u(t-2)-u(t-3)] + r(4-t)[u(t-3)-u(t-4)]
def f_3(t):
    y = my_ramp(t-2)*(my_step(t-2) - my_step(t-3)) + my_ramp(4-t)*(my_step(t-3) - my_step(t-4)) 
    return y

plt.subplot(3, 1, 3)
plt.plot(t,f_3(t))
plt.ylabel('f3(t)')
plt.ylim([0,1.2])
plt.xlabel('t')
plt.grid()
plt.show()


"""
Part 2 : Convolution
"""
def my_conv(f1,f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1Extended = np.append(f1,np.zeros((1,Nf2-1)))
    f2Extended = np.append(f2,np.zeros((1,Nf1-1)))
    result = np.zeros(f1Extended.shape)
    
    for i in range(Nf2 + Nf1 -2):      #Time inversion (i = t)
        result[i] = 0                  #Initialized
        for j in range(Nf1):           # j = tau (Integration)
             if(i-j + 1 > 0):          # If both Overlap
                 try:
                     result[i] += f1Extended[j]*f2Extended[i-j+1]    #x(tau)*h(t-tau)
                 except:
                    print(i ,j)         # If we errors
    return result

steps = 1e-2
t = np.arange(0, 20+steps, steps)
NN = len(t)
tExtended = np.arange(0, 2*t[NN-1], steps)   # For Convolution


# Convlovef_1(t) and f_2(t)
conv12 = my_conv(f_1(t), f_2(t))
conv12Check = sig.convolve(f_1(t), f_2(t))*steps

plt.figure(figsize=(10,7))
plt.title(' Convolution of f1(t) and f2(t)')
plt.figure(figsize=(10,7))
plt.title( 'Convolution of f1(t) and f2(t) ') 
plt.plot(tExtended, conv12, label= 'our convolution ')
plt.plot(tExtended, conv12Check, '--', label='Built-in convolution')
plt.ylabel('f1(t) * f2(t)')    
plt.xlabel('t') 
plt.ylim([0, 1.2])    
plt.grid()    
plt.legend( )    
plt.show()
 
# Convolve f_2(t) and f_3(t)
conv12 = my_conv(f_2(t), f_3(t))
conv12Check = sig.convolve(f_2(t), f_3(t))*steps

plt.figure(figsize=(10,7))
plt.title(' Convolution of f2(t) and f3(t)')
plt.figure(figsize=(10,7))
plt.title( 'Convolution of f2(t) and f3(t) ') 
plt.plot(tExtended, conv12, label= 'our convolution ')
plt.plot(tExtended, conv12Check, '--', label='Built-in convolution')
plt.ylabel('f2(t) * f3(t)')    
plt.xlabel('t') 
plt.ylim([0, 1.2])    
plt.grid()    
plt.legend( )    
plt.show()

# Convolve f_1(t) and f_3(t)
conv12 = my_conv(f_1(t), f_3(t))
conv12Check = sig.convolve(f_1(t), f_3(t))*steps

plt.figure(figsize=(10,7))
plt.title(' Convolution of f1(t) and f3(t)')
plt.figure(figsize=(10,7))
plt.title( 'Convolution of f1(t) and f3(t) ') 
plt.plot(tExtended, conv12, label= 'our convolution ')
plt.plot(tExtended, conv12Check, '--', label='Built-in convolution')
plt.ylabel('f1(t) * f3(t)')    
plt.xlabel('t') 
plt.ylim([0, 1.2])    
plt.grid()    
plt.legend( )    
plt.show()

 




