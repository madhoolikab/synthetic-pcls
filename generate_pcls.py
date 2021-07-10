# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 12:13:08 2021

@author: Madhu
"""
#%%
import os
os.chdir('D:/Lab/STC_DEMInterpol/synthetic_data/')

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#%%
centre = [-2, -2, 0]
r = 10.0
pi = np.pi
cos = np.cos
sin = np.sin
N = 100j
phi, theta = np.mgrid[0:pi/2:N, 0:2*pi:N]

x = centre[0] + r*sin(phi)*cos(theta)
y = centre[1] + r*sin(phi)*sin(theta)
z = centre[2] + r*cos(phi)

hemisphere_pcl = np.reshape([x,y], (2, -1)).T
hemisphere_z = z.flatten()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, c=z, s=1)

# ax.scatter(hemisphere_pcl[:, 0], hemisphere_pcl[:, 1], hemisphere_z, c=hemisphere_z, s=1)

#%%
''' Pyramid '''
def pyramid(n):
    r = np.arange(n)
    d = np.minimum(r,r[::-1])
    return np.minimum.outer(d,d)

n = 50

x = np.linspace(4, 14, n)
y = np.linspace(10, 15, n)
X, Y = np.meshgrid(x, y)
pyramid_pcl = np.meshgrid(x, y)
grid_shape = pyramid_pcl[0].shape
pyramid_pcl = np.reshape(pyramid_pcl, (2, -1)).T
Z = pyramid(n)
pyramid_z = Z.flatten()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
ax.scatter(X, Y, Z, c=Z, s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#%%

n = 75

x = np.linspace(4, 6, n)
y = np.linspace(10, 12, n)
X, Y = np.meshgrid(x, y)
pyramid_pcl_1 = np.meshgrid(x, y)
pyramid_pcl_1 = np.reshape(pyramid_pcl_1, (2, -1)).T
Z = pyramid(n)
pyramid_z_1 = Z.flatten()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X, Y, Z, c=Z, s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

n = 51

x = np.linspace(10, 14, n)
y = np.linspace(0, 6, n)
X, Y = np.meshgrid(x, y)
pyramid_pcl_2 = np.meshgrid(x, y)
pyramid_pcl_2 = np.reshape(pyramid_pcl_2, (2, -1)).T
Z = pyramid(n)
pyramid_z_2 = Z.flatten()

ax = fig.gca(projection='3d')
ax.scatter(X, Y, Z, c=Z, s=1)

n = 101

x = np.linspace(-7, -10, n)
y = np.linspace(-1, -6, n)
X, Y = np.meshgrid(x, y)
pyramid_pcl_3 = np.meshgrid(x, y)
pyramid_pcl_3 = np.reshape(pyramid_pcl_3, (2, -1)).T
Z = pyramid(n)
pyramid_z_3 = Z.flatten()

ax = fig.gca(projection='3d')
ax.scatter(X, Y, Z, c=Z, s=1)

all_pcl = np.vstack((pyramid_pcl_1, pyramid_pcl_2, pyramid_pcl_3))
all_Z = np.hstack((pyramid_z_1, pyramid_z_2, pyramid_z_3))
all_pcl_data = np.c_[all_pcl, all_Z]
# np.save('three_pyramids.npy', all_pcl_data)

#%%
all_pcl = np.vstack((hemisphere_pcl, pyramid_pcl))
all_Z = np.hstack((hemisphere_z, pyramid_z))

all_pcl_data = np.c_[all_pcl, all_Z]
# np.save('_denser_hemisphere_and_pyramid.npy', all_pcl_data)
# loaded_data = np.load('syn_data_1.npy')
#%%
''' Data of all the shapes in one palce '''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(all_pcl[:, 0], all_pcl[:, 1], all_Z, s=1, c=all_Z)
# ax.scatter(all_pcl[:, 0], all_pcl[:, 1], all_Z, c=all_Z, s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#%%
base_x = np.random.default_rng().uniform(min(all_pcl[:, 0])-3,max(all_pcl[:, 0])+3,1000)
base_y = np.random.default_rng().uniform(min(all_pcl[:, 1])-3,max(all_pcl[:, 1])+3,1000)
base_depth = np.random.default_rng().uniform(-10.0,-9.0,1000)



# fig = plt.figure()
# ax = fig.gca(projection='3d')
ax.scatter(base_x, base_y, base_depth, c=base_depth, s=1)
# ax.set_zlim3d(-15, -5)

#%%
r = base_x > np.min(hemisphere_pcl[:, 0])

s = base_x > np.max(hemisphere_pcl[:, 0])

p = r & s

p = ~p
 
ind = p.nonzero()

new_base_x = np.delete(base_x, ind)

new_base_y = np.delete(base_y, ind)

new_base_depth = np.delete(base_depth, ind)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(all_pcl[:, 0], all_pcl[:, 1], all_Z, s=1, c=all_Z)
# ax.scatter(all_pcl[:, 0], all_pcl[:, 1], all_Z, c=all_Z, s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.scatter(new_base_x, new_base_y, new_base_depth, c=new_base_depth, s=1)

#%%
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np

dem = np.load('jacksboro_fault_dem.npz')

#%%
z = dem['elevation']
nrows, ncols = z.shape
x = np.linspace(dem['xmin'], dem['xmax'], ncols)
y = np.linspace(dem['ymin'], dem['ymax'], nrows)
x, y = np.meshgrid(x, y)

region = np.s_[5:65, 5:65]
x, y, z = x[region], y[region], z[region]
pcl = np.reshape([x, y], (2, -1)).T
pcl_z = z.flatten()

#%%
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

ls = LightSource(0,0)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
# surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=False, shade=False)
ax.scatter(pcl[:, 0], pcl[:, 1], z, c=z, s=1)

#%%
''' pcl in stc report - parameters are as below 

centre = [-2, -2, 0]
r = 10.0
N = 100j

n = 50

x = np.linspace(4, 14, n)
y = np.linspace(10, 15, n)

'''


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax.plot_surface(x, y, z)

plt.show()












