import numpy as np
from scipy.spatial import cKDTree

class tree(object):
    def __init__(self, X=None, z=None, leafsize=10):
        print('init')
        if not X is None:
            self.tree = cKDTree(X, leafsize=leafsize )
            self.original_locations = np.copy(X)
        if not z is None:
            self.z = np.array(z)

    def fit(self, X=None, z=None, leafsize=10):
        print('fit')
        return self.__init__(X, z, leafsize)

    def __call__(self, X, r=1.0, p=2, eps=1e-6, regularize_by=1e-9):
        print('func: call, radius =',r)
        full_grid_tree = cKDTree(X)
        self.neighbours_indexes = full_grid_tree.query_ball_tree(self.tree, r, eps=eps, p=p)
        idw_values = []
        for point, neighbours_locs in zip(X, self.neighbours_indexes):
            neighbours = self.original_locations[neighbours_locs]
            dist = np.linalg.norm(point-neighbours, axis=1) + regularize_by
            wt = 1/(dist**2)
            idw_values.append(np.sum(self.z[neighbours_locs] * wt)/np.sum(wt))
        
                
        return idw_values #, pts, nbrs
             
    def idw_interpolation(self, X, r=1.0, p=2, eps=1e-6, regularize_by=1e-9):
        print('func: idw_interpolation, radius =',r)
        full_grid_tree = cKDTree(X)
        self.neighbours_indexes = full_grid_tree.query_ball_tree(self.tree, r, eps=eps, p=p)
        idw_values = []
        
        for point, neighbours_locs in zip(X, self.neighbours_indexes):
            neighbours = self.original_locations[neighbours_locs]
            dist = np.linalg.norm(point-neighbours, axis=1) + regularize_by
            wt = 1/(dist**2)
            idw_values.append(np.sum(self.z[neighbours_locs] * wt)/np.sum(wt))
            
        return idw_values
    
    def gaussian_interpolation(self, X, r=1.0, p=2, eps=1e-6, regularize_by=1e-9):
        print('func: gaussian_interpolation, radius =',r)
        full_grid_tree = cKDTree(X)
        self.neighbours_indexes = full_grid_tree.query_ball_tree(self.tree, r, eps=eps, p=p)
        gaussian_values = []
        
        for point, neighbours_locs in zip(X, self.neighbours_indexes):
            neighbours = self.original_locations[neighbours_locs]
            dist = np.linalg.norm(point-neighbours, axis=1)
            gaussian_wt = np.exp(-1*(dist)**2)
            gaussian_values.append(np.sum(self.z[neighbours_locs] * gaussian_wt)/np.sum(gaussian_wt))
            
        return gaussian_values
    
    def bilateral_interpolation(self, X, X_depths, r=1.0, p=2, eps=1e-6, regularize_by=1e-9):
        print('blf interpolation')
        full_grid_tree = cKDTree(X)
        self.neighbours_indexes = full_grid_tree.query_ball_tree(self.tree, r, eps=eps, p=p)
        bilateral_values = []
        sigma_d = 1.0
        sigma_r = 1.0
        for point, init_depth, neighbours_locs in zip(X, X_depths, self.neighbours_indexes):
            neighbours = self.original_locations[neighbours_locs]
            dist = np.linalg.norm(point-neighbours, axis=1)
            domain_wt = np.exp((-1*(dist)**2)/2*sigma_d*sigma_d)
            depth_dist = (init_depth - self.z[neighbours_locs])**2
            range_wt = np.exp((-1*(depth_dist)**2)/2*sigma_r*sigma_r)
            bilateral_values.append(np.sum(self.z[neighbours_locs]*domain_wt*range_wt)/np.sum(domain_wt*range_wt))
            
        return bilateral_values
    
    def new_bilateral_interpolation(self, X, X_depths, sigma_d=1.0, sigma_r=1.0, r=1.0, p=2, eps=1e-6, regularize_by=1e-9):
        print('blf interpolation, r = {}, sigma_d = {}, sigma_r = {}'.format(r, sigma_d, sigma_r))
        full_grid_tree = cKDTree(X)
        self.neighbours_indexes = full_grid_tree.query_ball_tree(self.tree, r, eps=eps, p=p)
        bilateral_values = []
        for point, init_depth, neighbours_locs in zip(X, X_depths, self.neighbours_indexes):
            neighbours = self.original_locations[neighbours_locs]
            dist = np.linalg.norm(point-neighbours, axis=1)
            domain_wt = np.exp((-1*(dist)**2)/2*sigma_d*sigma_d)
            depth_dist = (init_depth - self.z[neighbours_locs])**2
            range_wt = np.exp((-1*(depth_dist)**2)/2*sigma_r*sigma_r)
            bilateral_values.append(np.sum(self.z[neighbours_locs]*domain_wt*range_wt)/np.sum(domain_wt*range_wt))
            
        return bilateral_values
        
    def transform(self, X, k=6, p=2, eps=1e-6, regularize_by=1e-9):
        print('transform')
        return self.__call__(X, k, eps, p, regularize_by)

def demo():
    import matplotlib.pyplot as plt

    # create sample points with structured scores
    X1 = 10 * np.random.rand(1000, 2) -5

    def func(x, y):
        return np.sin(x**2 + y**2) / (x**2 + y**2)

    z1 = func(X1[:,0], X1[:,1])

    # 'train'
    idw_tree = tree(X1, z1)

    # 'test'
    spacing = np.linspace(-5., 5., 100)
    X2 = np.meshgrid(spacing, spacing)
    grid_shape = X2[0].shape
    X2 = np.reshape(X2, (2, -1)).T
    z2 = idw_tree(X2)

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10,3))
    ax1.contourf(spacing, spacing, func(*np.meshgrid(spacing, spacing)))
    ax1.set_title('Ground truth')
    ax2.scatter(X1[:,0], X1[:,1], c=z1, linewidths=0)
    ax2.set_title('Samples')
    ax3.contourf(spacing, spacing, z2.reshape(grid_shape))
    ax3.set_title('Reconstruction')
    plt.show()
    return
