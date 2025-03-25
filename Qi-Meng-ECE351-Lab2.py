#%% Background
import numpy as np
import matplotlib . pyplot as plt

plt.rcParams.update ({'font.size': 14}) # Set font size in plots

steps = 1e-2 # Define step size
t = np . arange (0 , 5 + steps , steps ) 
# Add a step size to make sure the plot includes 5.0.
#Since np.arange() only goes up to,but doesn't include the value of the second argument


print ('Number of elements: len(t) = ', len(t) , '\nFirst Element: t[0] = ', t [0] , 
     '\nLast Element: t[len(t) - 1] = ', t [len(t)-1])
# Notice the array might be a different size than expected since Python starts at 0.
# Then we will use our knowledge of indexing to have Python print the first and last index of the array.
#Notice the array goes from 0 to len()-1

# --- User - Defined Function ---

# Create output y(t) using a for loop and if/ else statements
def example1 (t) : # The only variable sent to the function is t
    y = np.zeros (t.shape) # initialze y(t) as an array of zeros

    for i in range (len(t)) : # run the loop once for each index of t
        if i < (len(t)+1) /3:
            y [i] = t [i]**2
        else :
            y [i] = np . sin (5*t [i]) + 2
    return y 

y = example1 (t) # call the function we just created
plt . figure (figsize = (10, 7) )
plt . subplot (2, 1, 1)
plt . plot (t, y )
plt . grid ()
plt . ylabel ('y(t) with Good Resolution')
plt . title ('Background - Illustration of for Loops and if/ else Statements')

t = np . arange (0, 5+0.25, 0.25) # redefine t with poor resolution
y = example1 (t)

plt . subplot (2 , 1 , 2)
plt . plot (t , y)
plt . grid ()
plt . ylabel ('y(t) with Poor Resolution')
plt . xlabel ('t')
plt . show ()
#%% Part 2  Task 2
#   ------ hand derived equation ------
#   y(t) = r(t) - r(t-3) + 5u(t-3) -2u(t-6) - 2r(t-6)
import numpy as np
import matplotlib.pyplot as plt

def my_step(t):
    y = np.zeros((len(t), 1))
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = 1
        else:
            y[i] = 0
    return y

def my_ramp(t):
    y = np.zeros((len(t), 1))
    for i in range(len(t)):
        if t[i] >= 0:
            y[i] = t[i]
        else:
            y[i] = 0
    return y 

steps = 1e-3
t = np.arange(-4,4+steps,steps)
u = my_step(t)
r = my_ramp(t)

plt.figure(figsize=(10,7))
plt.subplot(2, 1, 1)
plt.plot(t, u)
plt.grid()
plt.ylabel('u(t)')
plt.xlabel('t')
plt.title('Step and Ramp Function')
plt.ylim([-1, 1.5])
plt.subplot(2, 1, 2)
plt.plot(t, r)
plt.ylabel('r(t)')
plt.xlabel('t')
plt.ylim([-1, 5])
plt.grid()
plt.show()

def func2(t):
    return(my_ramp(t)-my_ramp(t-3)+5*my_step(t-3)-2*my_step(t-6)-2*my_ramp(t-6))
    
steps = 1e-2
t = np.arange(-5, 10+steps, steps)

y = func2(t)

plt.figure(figsize=(10,7))
plt.plot(t, y)
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Plot of function')
plt.grid()
plt.show()

#%% Part 3  Task 1
steps = 1e-2
t = np.arange(-10, 5+steps, steps)

y=func2(-t)

plt.figure(figsize=(10,7))
plt.plot(t, y)
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Time Reversal Function')
plt.grid()
plt.show()

#%% Part 3  Task 2
steps = 1e-2
t = np.arange(-15, 15+steps, steps)
y1 = func2(t-4)
y2 = func2(-t-4)
plt.figure(figsize=(10,7))
plt.plot(t, y1, label='f(t-4)')
plt.plot(t, y2, label='f(-t-4)')
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('The result of Time-Shift Operation')
plt.grid()
plt.legend()
plt.show()

#%% Part 3  Task 3
steps = 1e-2
t = np.arange(-14, 20+steps, steps)

y1 = func2(2*t)
y2 = func2(t/2)

plt.figure(figsize=(10,7))
plt.plot(t, y1, label='f(2t)')
plt.plot(t, y2, label='f(t/2)')
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('The result of Time Scale Operation')
plt.grid()
plt.legend()
plt.ylim([-2, 10])
plt.show()