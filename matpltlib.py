# Python program using Matplotlib 
# for forming a linear plot 

# importing the necessary packages and modules 
import matplotlib.pyplot as plt 
import numpy as np 

# Prepare the data 
x = np.linspace(60,300,2000)
y = np.linspace(1,10,2000)

# Plot the data 
plt.plot(x, y, label ='Nisahnt') 

# Add a legend 
plt.legend() 

# Show the plot 
plt.show() 
