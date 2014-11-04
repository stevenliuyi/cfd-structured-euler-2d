import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def read_data():
    filename = 'data.txt'
    f   = open(filename)
    x   = np.zeros((mx-1, my-1))
    y   = np.zeros((mx-1, my-1))
    rho = np.zeros((mx-1, my-1))
    u   = np.zeros((mx-1, my-1))
    v   = np.zeros((mx-1, my-1))
    et  = np.zeros((mx-1, my-1))
    m   = np.zeros((mx-1, my-1))
    points = []
    mm = []
    i = 0; j = 0
    for line in f:
        x[i,j]   = float(line.split()[0])
        y[i,j]   = float(line.split()[1])
        rho[i,j] = float(line.split()[2])
        u[i,j]   = float(line.split()[3]) / float(line.split()[2])
        v[i,j]   = float(line.split()[4]) / float(line.split()[2])
        et[i,j]  = float(line.split()[5])
        m[i,j]   = float(line.split()[6])

        if (j == my-2):
            i += 1; j = 0
        else:
            j += 1
        if (i == mx-1): break

    f.close()
    return (x,y,rho,u,v,et,m)
    
# grid points
mx = 65; my = 17 
mx = 129; my = 33

gamma = 1.4
# read results
(x,y,rho,u,v,et,m) = read_data()
#print y.transpose()

# pressure
p = (gamma-1)*(et - .5*rho*(u**2+v**2))

# entrophy
s = gamma/(gamma-1) * np.log(p/rho) - np.log(p)

fig = plt.figure()
ax  = fig.add_subplot(111, aspect='equal')

contour = ax.contourf(x,y,m,15,cmap=plt.cm.rainbow)
plt.colorbar(contour, orientation='horizontal')


ax.set_xlim(-1,2)
ax.set_ylim(0,1)
plt.show()

