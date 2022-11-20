

# Fixing random state for reproducibility
np.random.seed(19680801)

method_list=['Cyclic Coordinate Descent','Powell','Hooke Jeeves','Nelder Mead']
import numpy as np
import matplotlib.pyplot as pp
val = 0. # this is the value where you want the data to appear on the y-axis.
ar = np.arange(4) # just as an example array
fig = pp.figure(figsize=(6.4,2.1),frameon=False)
ax = fig.add_subplot(111)
xs = np.linspace(23,32,4)
ys = np.linspace(0,0,4)
for i in range(4):

    l1 = np.array((xs[i], ys[i]))

    th2 = ax.text(*l1, method_list[i], fontsize=16,
              rotation=45, rotation_mode='anchor')


    ax.scatter(xs, ys,  s=300,c = 'b')
    i = i + 1  

ax.set_xlabel('(detailed)                                 Level of abstraction                                (visionary)')
ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_ticks([])

pp.show()