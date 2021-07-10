# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 06:52:05 2021

@author: Madhu
"""
#%%
import os
os.chdir('D:/Lab/STC_DEMInterpol/synthetic_data/')

#%%
import matplotlib.pyplot as plt
import numpy as np
import idw
from skimage.metrics import structural_similarity as ssim
from numpy.random import default_rng
from matplotlib import cm
import matplotlib.animation as animation
from scipy.interpolate import griddata

#%%
def mse(ground_truth, reconstruction):
    return np.mean((ground_truth - reconstruction)**2)
    
#%%
def snr(r, t):
    r_squared = r**2
    error_squared = (r - t)**2
    return 10*np.log10(np.sum(r_squared, axis=0)/np.sum(error_squared, axis=0))

#%%
def psnr(ground_truth, reconstruction):
    M = np.max(ground_truth)
    return 10*np.log10(M**2/(mse(ground_truth, reconstruction)))

#%%
# syn_pcl = np.load('three_pyramids.npy')
# syn_pcl = np.load('part_of_pcl1.npy')
syn_pcl = np.load('_denser_hemisphere_and_pyramid.npy')

#%%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(syn_pcl[:, 0], syn_pcl[:, 1], syn_pcl[:, 2], s=1, c=syn_pcl[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#%%
rows, cols = syn_pcl.shape
data_points_percent = 0.1
data_count = int(rows*data_points_percent)

rng = default_rng(0)
data_indices = rng.choice(rows, size=data_count, replace=False)

np.random.seed(0)
syn_pcl_with_noise = syn_pcl + np.c_[np.random.normal(0, 0.3, (rows, cols-1)), np.zeros(rows)]

observed_pcl = syn_pcl[data_indices]
# observed_pcl = np.load('syn_2_observed_pcl.npy')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(observed_pcl[:, 0], observed_pcl[:, 1], observed_pcl[:, 2], s=3, c=observed_pcl[:, 2])
# ax.scatter(syn_pcl_with_noise[:100, 0], syn_pcl_with_noise[:100, 1], syn_pcl_with_noise[:100, 2], s=3, c=syn_pcl_with_noise[:100, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#%%
x = observed_pcl[:, 0]
y = observed_pcl[:, 1]
X = np.copy(observed_pcl[:, :2])
z = np.copy(observed_pcl[:, 2]) # depth values

min_x, max_x = min(x), max(x)
min_y, max_y = min(y), max(y)
min_z, max_z = min(z), max(z)

n = 50j
box_x, box_y = np.mgrid[min_x:max_x:n, min_y:max_y:n]
spacing_x = np.linspace(min_x, max_x, int(n.imag))
spacing_y = np.linspace(min_y, max_y, int(n.imag))

full_pcl = np.meshgrid(spacing_x, spacing_y)
grid_shape = full_pcl[0].shape
full_pcl = np.reshape(full_pcl, (2, -1)).T

full_pcl = np.copy(syn_pcl[:, :2])
ground_truth = np.copy(syn_pcl[:, 2])

#%%
pcl_idw_tree = idw.tree(X, z)
p_val=2.0
pcl = 1
radius = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0] #, 12.0, 12.5, 15.0, 20.0]
# radius = [5.0, 7.5, 10.0, 12.0, 12.5, 15.0, 20.0]

#%%
syn_idw_results = []
for val in radius:
    z_full = pcl_idw_tree(full_pcl, r=val, p=p_val)
    z_final = np.array(z_full)#.reshape(grid_shape)
    # z_final_T = z_final.T
    syn_idw_results.append(z_final)

#%%
i=0
for res in syn_idw_results:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(full_pcl[:, 0], full_pcl[:, 1], res, s=1, c=res)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('p={}, radius={}'.format(p_val, radius[i]))
    i += 1

#%%
for res in syn_idw_results:
    print(mse(ground_truth, res), psnr(ground_truth, res))

#%%
syn_gaussian_results = []
for val in radius:
    z_full = pcl_idw_tree.gaussian_interpolation(full_pcl, r=val, p=p_val)
    z_final = np.array(z_full) #.reshape(grid_shape)
    # z_final_T = z_final.T
    syn_gaussian_results.append(z_final)

#%%
i=0
for res in syn_gaussian_results:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(full_pcl[:, 0], full_pcl[:, 1], res, s=1, c=res)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('p={}, radius={}'.format(p_val, radius[i]))
    i += 1

#%%
for res in syn_gaussian_results:
    print(mse(ground_truth, res), psnr(ground_truth, res))
    
#%%
# grid_linear = griddata(observed_pcl[:, :2], observed_pcl[:, 2], (box_x, box_y), method='linear')
grid_linear = griddata(observed_pcl[:, :2], observed_pcl[:, 2], full_pcl, method='linear')

masked_linear = np.ma.masked_where(np.isnan(grid_linear), grid_linear)
fig2 = plt.figure(figsize=(15,10))
ax2 = plt.axes(projection='3d')
# surf2 = ax2.plot_surface(box_x, box_y, masked_linear, cmap=cm.coolwarm, vmin=np.min(masked_linear), vmax=np.max(masked_linear))
# surf2 = ax2.plot_trisurf(full_pcl[:, 0], full_pcl[:, 1], masked_linear, cmap=cm.coolwarm, vmin=np.min(masked_linear), vmax=np.max(masked_linear))
ax2.scatter(full_pcl[:, 0], full_pcl[:, 1], masked_linear, cmap=cm.jet, s=1, vmin=np.min(masked_linear), vmax=np.max(masked_linear))
ax2.scatter(x, y, z, c=z, s=1)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Linear Interpolation')
# fig2.colorbar(surf2)

full_pcl_init_depths = masked_linear.flatten(order='F')

#%%
syn_blf_results = []
for val in radius:
    print(val)
    z_full = pcl_idw_tree.bilateral_interpolation(full_pcl, full_pcl_init_depths, r=val, p=p_val)
    z_final = np.array(z_full) #.reshape(grid_shape)
    # z_final_T = z_final.T
    syn_blf_results.append(z_final)

#%%
for res in syn_blf_results:
    masked_res = np.ma.masked_where(np.isnan(res), res)
    print(mse(ground_truth, masked_res), psnr(ground_truth, masked_res))
    
#%%
i=0
for res in syn_blf_results:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(full_pcl[:, 0], full_pcl[:, 1], res, s=1, c=res)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('p={}, radius={}'.format(p_val, radius[i]))
    i = i+1
    
#%%
custom_syn_blf_results = []
sigma_d = 1.0
sigma_r = 0.1
for val in radius:
    z_full = pcl_idw_tree.new_bilateral_interpolation(full_pcl, full_pcl_init_depths, sigma_d, sigma_r, r=val, p=p_val)
    z_final = np.array(z_full) #.reshape(grid_shape)
    # z_final_T = z_final.T
    custom_syn_blf_results.append(z_final)

#%%
for res in custom_syn_blf_results:
    masked_res = np.ma.masked_where(np.isnan(res), res)
    print(mse(ground_truth, masked_res), psnr(ground_truth, masked_res))
    
#%%
i=0
for res in custom_syn_blf_results:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(full_pcl[:, 0], full_pcl[:, 1], res, s=1, c=res)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('p={}, radius={}'.format(p_val, radius[i]))
    i = i+1

#%%
def display_gif(results, p, radius, title):
    writergif = animation.PillowWriter(fps=0.8)
    
    def animate2(i):
        print('animate', i)
        ax.clear()
        masked_res = np.ma.masked_where(np.isnan(results[i]), results[i])
        ax.scatter(full_pcl[:, 0], full_pcl[:, 1], masked_res, c=masked_res, s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title + ' with p={}, radius={}'.format(2, radius[i]))
    
    
    fig = plt.figure(figsize=(15,10))
    ax = plt.axes(projection='3d')
    print(len(results))
    ani = animation.FuncAnimation(fig, animate2, frames=range(0, len(results)), interval=1000)
    
    plt.show()

display_gif(custom_syn_blf_results, p_val, radius, 'sigma_d = {}, sigma_r = {}'.format(sigma_d, sigma_r))
#%%
# syn_idw_results = np.load('syn_2_idw_results.npy')
# syn_idw_results = list(syn_idw_results)

# syn_gaussian_results = np.load('syn_2_gaussian_results.npy')
# syn_gaussian_results = list(syn_gaussian_results)

#%%
writergif = animation.PillowWriter(fps=0.8) 

def animate2(i):
    print('animate', i)
    ax.clear()
    masked_res = np.ma.masked_where(np.isnan(syn_idw_results[i]), syn_idw_results[i])
    # ax.plot_surface(box_x, box_y, masked_res, cmap=cm.coolwarm, vmin=np.min(masked_res), vmax=np.max(masked_res))
    # ax.scatter(x, y, z, c=z, s=1)
    ax.scatter(full_pcl[:, 0], full_pcl[:, 1], masked_res, c=masked_res, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.set_title('Inverse Distance Weighted with p={}, radius={}'.format(2, radius[i]))
    

fig = plt.figure(figsize=(20,10))
ax = plt.axes(projection='3d')
ani = animation.FuncAnimation(fig, animate2, frames=range(0, len(syn_idw_results)), interval=1000)

plt.show()

#%%
writergif = animation.PillowWriter(fps=0.8) 

# elev = 13.241 # 31.44
# azim = 45.677 # -64.46

# # for pcl 4 - 
# elev = 31.44 
# azim = 64.46

def animate2(i):
    print('animate', i)
    ax.clear()
    masked_res = np.ma.masked_where(np.isnan(syn_gaussian_results[i]), syn_gaussian_results[i])
    # ax.plot_surface(box_x, box_y, masked_res, cmap=cm.coolwarm, vmin=np.min(masked_res), vmax=np.max(masked_res))
    # ax.scatter(x, y, z, c=z, s=1)
    ax.scatter(full_pcl[:, 0], full_pcl[:, 1], masked_res, c=masked_res, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.set_title('Using Gaussian Weights with p={}, radius={}'.format(2, radius[i]))
    # ax.set_title('Inverse Distance Weighted with p={}, radius={}'.format(2, radius[i]))
    # ax.view_init(elev, azim)
    

fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection='3d')
# animate2(0)
ani = animation.FuncAnimation(fig, animate2, frames=range(0, len(syn_gaussian_results)), interval=1000)
# ani.save('syn_2_idw_' + str(data_points_percent) + '_pcl_' + str(pcl) + '_with_radius_gif.gif', writer=writergif)
# ani.save('syn_gaussian_pcl' + str(pcl) + '_full_size_with_radius_gif.gif', writer=writergif)


# ax.view_init(elev, azim)

plt.show()

print('ax.azim {}'.format(ax.azim))
print('ax.elev {}'.format(ax.elev))



#%%
from numpy import inf
from numpy import nan
writergif = animation.PillowWriter(fps=0.8) 

# elev = 41.76031360836282 # 31.44
# azim = 29.209308564147136 # -64.46

def animate2(i):
    print('animate', i)
    res = syn_blf_results[i]
    ax.clear()
    # res[res == -inf] = nan
    # res[res == inf] = nan
    masked_res = np.ma.masked_where(np.isnan(res), res)
    ax.scatter(full_pcl[:, 0], full_pcl[:, 1], masked_res, c=masked_res, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.set_title('BLF Interpolation with p={}, radius={}'.format(2, radius[i]))
    # ax.view_init(elev, azim)
    

fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection='3d')
ani = animation.FuncAnimation(fig, animate2, frames=range(0, len(syn_blf_results)), interval=1000)
# ani.save('new_blf_pcl' + str(pcl) + '_with_radius_gif.gif', writer=writergif)

plt.show()

# print('ax.elev {}'.format(ax.elev))
# print('ax.azim {}'.format(ax.azim))

#%%
writergif = animation.PillowWriter(fps=0.8) 

def animate2(i):
    print('animate', i)
    res = custom_syn_blf_results[i]
    ax.clear()
    masked_res = np.ma.masked_where(np.isnan(res), res)
    ax.scatter(full_pcl[:, 0], full_pcl[:, 1], masked_res, c=masked_res, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.set_title('BLF Interpolation with p={}, radius={}'.format(2, radius[i]))
    

fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection='3d')
ani = animation.FuncAnimation(fig, animate2, frames=range(0, len(custom_syn_blf_results)), interval=1000)
# ani.save('new_blf_pcl' + str(pcl) + '_with_radius_gif.gif', writer=writergif)

plt.show()

#%%










#%%
writergif = animation.PillowWriter(fps=0.8) 

def animate2(i):
    print('animate', i)
    ax.clear()
    masked_res = np.ma.masked_where(np.isnan(syn_idw_results[i]), syn_idw_results[i])
    ax.plot_surface(box_x, box_y, masked_res, cmap=cm.coolwarm, vmin=np.min(masked_res), vmax=np.max(masked_res))
    ax.scatter(x, y, z, c=z, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.set_title('Inverse Distance Weighted with p={}, radius={}'.format(2, radius[i]))
    

fig = plt.figure(figsize=(20,10))
ax = plt.axes(projection='3d')
ani = animation.FuncAnimation(fig, animate2, frames=range(0, len(syn_idw_results)), interval=1000)

plt.show()










