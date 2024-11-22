import quapy as qp
import quapy.functional as F
from scipy.stats import dirichlet, chi2
import numpy as np

from confidence import ConfidenceRegionSimplex

# from model import ConfidenceRegion, IdentityFunction


#points = F.uniform_simplex_sampling(n_classes=3, size=5)
num_points = 50

simplex_dim = 2
alphas = [8,1,10]
points = dirichlet.rvs(alpha=alphas[:simplex_dim+1], size=num_points, random_state=0)
print(points)

cr = ConfidenceRegionSimplex(points)


# checks whether all points in the surface of the ellipse are contained in the probability simplex
num_samples=100

# generates num_samples points in the surface of a sphere
np.random.seed(1)
points_on_sphere = np.random.normal(size=(num_samples, simplex_dim))
points_on_sphere /= np.linalg.norm(points_on_sphere, axis=1)[:, np.newaxis]
points_on_sphere = np.hstack([np.zeros(shape=(points_on_sphere.shape[0],1)), points_on_sphere])
print('points on sphere')
print(points_on_sphere)
print('with l2 norm', np.linalg.norm(points_on_sphere, axis=1))

axes = cr.semiaxes
print('axes')
print(axes)
#_, eigenvalues, eigenvectors = cr.reduced_cov()
eigenvalues, eigenvectors = np.linalg.eigh(cr.cov)
small_eigenvalues = eigenvalues < 1e-7
eigenvalues[small_eigenvalues]=0
chi2_critical = chi2.ppf(0.95, df=simplex_dim)
axes = np.sqrt(eigenvalues * chi2_critical)
print('ax')
print(axes)
print('eigenvectors')
print(eigenvectors)
print('eigenvalues')
print(eigenvalues)

# scale by ellipse's axes
scaled_points = points_on_sphere * axes
print('scaled points')
print(scaled_points)

# rotate
ellipsoidal_points = scaled_points.dot(eigenvectors.T)  # Rotar los puntos
print('elipsoidal points')
print(ellipsoidal_points)

# traslate to mean
ellipsoidal_points += cr.mean()
print('elipsoidal points traslated')
print(ellipsoidal_points)

all_positive = (ellipsoidal_points>0).all()
sum_1 = np.isclose(ellipsoidal_points.sum(axis=1), 1).all()
all_in_simplex = all_positive and sum_1
print(f'{all_positive=}')
print(f'{sum_1=}')
print(f'{all_in_simplex=}')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

points = ellipsoidal_points

# Crear la figura y el eje 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Desempaquetar los puntos en coordenadas x, y, z
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# Graficar los puntos
ax.scatter(x, y, z, c='b', marker='o')  # Puedes cambiar el color y el marcador
mu = cr.mean()
ax.scatter(mu[0], mu[1], mu[2], c='r', marker='x')  # Puedes cambiar el color y el marcador

# Punto de origen para ambos vectores
origin = mu # np.array([0, 0, 0])
print(origin)
# Graficar los vectores
ax.quiver(*origin, *eigenvectors[:,1]*axes[1], color='r', label='eigenvector 1', arrow_length_ratio=0.1)
ax.quiver(*origin, *eigenvectors[:,2]*axes[2], color='b', label='eigenvector 2', arrow_length_ratio=0.1)
#ax.quiver(*origin, *(eigenvectors[:,1]*ax[1]), color='r', label='eigenvector 1', arrow_length_ratio=0.1)
#ax.quiver(*origin, *(eigenvectors[:,2]*ax[2]), color='b', label='eigenvector 2', arrow_length_ratio=0.1)

# Crear el hiperplano usando los puntos (0,0,1), (0,1,0), y (1,0,0)
plane_points = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

# Generar una cuadrícula de puntos en el plano
x_plane = np.linspace(0, 1, 10)
y_plane = np.linspace(0, 1, 10)
x_grid, y_grid = np.meshgrid(x_plane, y_plane)

# Calcular z para el hiperplano usando la ecuación del plano
# Ecuación del plano: x + y + z = 1 -> z = 1 - x - y
z_plane = 1 - x_grid - y_grid

# Filtrar los puntos que están fuera del rango [0, 1]
z_plane[z_plane < 0] = np.nan  # Esto hará que esos puntos no se dibujen

# Graficar el hiperplano semitransparente
ax.plot_surface(x_grid, y_grid, z_plane, alpha=0.5, color='orange', edgecolor='none')

# Etiquetas de los ejes
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# Título
# ax.set_title('3D Point Representation')

ax.view_init(elev=10, azim=20)

plt.savefig('ellipse_out.pdf', bbox_inches='tight')
print('done')



