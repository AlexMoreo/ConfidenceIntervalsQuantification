import numpy as np
from scipy.special import digamma, polygamma

# Función para estimar alpha mediante MLE
def dirichlet_mle(X, tol=1e-7, max_iter=10000):
    """
    Estima los parámetros alpha para un conjunto de datos X.
    X es una matriz donde cada fila es una muestra (cada muestra en el simplexo).
    """
    # Inicialización de alpha
    n_samples, n_features = X.shape
    alpha_init = np.ones(n_features)
    alpha = alpha_init.copy()

    # Iteración de Newton-Raphson
    for iteration in range(max_iter):
        alpha_old = alpha.copy()
        g = digamma(np.sum(alpha)) - digamma(alpha) + np.mean(digamma(X), axis=0) - digamma(np.mean(X, axis=0))
        h = - polygamma(1, alpha)
        z = np.sum(h)
        b = np.sum(g / h) / (1/z + np.sum(1/h))
        alpha -= (g - b) / h

        if np.linalg.norm(alpha - alpha_old) < tol:
            break

    return alpha

# Ejemplo de uso
X = np.random.dirichlet([2, 5, 3], size=100)  # Simulamos 100 muestras de una Dirichlet
alpha_estimate = dirichlet_mle(X)
print("Estimación de alpha:", alpha_estimate)