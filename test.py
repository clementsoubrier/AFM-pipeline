
# %matplotlib qt5 # if interactive window outside the 
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np

x=np.array([[0,1,2,3,4],[0,1,2,3,3],[0,1,2,3,4]])
z=np.array([[0,1,5,3,2],[0,2,7,5,5],[8,13,5,9,2]])
y=np.array([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]])
plt.scatter(x,z)
# ax = plt.axes(projection='3d')
# ax.scatter(x,y,z)
# ax.plot_surface(x, y, z, cmap="cividis", lw=0.5, rstride=1,
#                         cstride=1, alpha=0.7, edgecolor='none',
#                         norm=mplc.PowerNorm(gamma=0.6))
plt.show()

test1=np.array([[0,1,2,3,4]])
test1[0,:]=1
print(test1)
print(np.concatenate((test1,test1)))