from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

method_list=['Nelder Mead','Hooke Jeeves','SGD','Genetic','PSO','GD']
def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


n = 7
i = 1
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for zlow, zhigh in [( -50, -25), ( -30, -5)]:
    
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.text(xs[i], ys[i], zs[i], method_list[i], color='red')
    ax.scatter(xs, ys, zs, c = 'b')
    i = i + 1  

ax.set_xlabel('Risk acceptance')
ax.set_ylabel('Level of confidence')
ax.set_zlabel('Problem dimentionality')

plt.show()