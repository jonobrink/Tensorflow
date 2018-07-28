import matplotlib.pyplot as plt

plt.fan

import numpy as np

sample = [np.random.normal(0,1,30) for x in range(30)]
iteration = list(range(10,310, 10))
iteration
sample[]
plt.plot
import ml_graph
from importlib import reload
reload(ml_graph)
help(ml_graph)

import numpy as np
dir(np)
help(np.histogram)

np.histogram(sample)
plt.hist(np.histogram(sample)
plt.plot(np.histogram(sample)[1][:10],np.histogram(sample, normed = True)[0][:10])

z = iteration
iteration = np.array(iteration)

blah = np.array([np.histogram(sample[x], normed = True) for x in range(30)])
x=blah[:,0]
y=blah[:,1]

x
y
z

x
y
x = sample[1]
y =





time = iteration
density = np.histogram
beta_values =

help(np.histogram)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
