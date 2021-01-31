import numpy as np
import matplotlib.pyplot as plt

from linear_kf import LinKF

if __name__ == '__main__':
    N = 100  # Number of observations / timeframes
    n = 2  # Dimension of the state vector X
    x = np.zeros((N, n))
    x[0] = [100, 20]
    a = 20  # Acceleration
    t = 1  # Difference in time
    A = np.array([[1, t], [0, 1]])  # Linear in x and v (state vector)
    B = np.array([[0.5 * t ** 2], [t]])  # Linear in a (control vector)
    u = np.array([a] * N)  # Controls vector

    rng = np.random.default_rng()

    # State transition error covariance matrix
    R = np.array([[400, 0], [0, 25]])

    # Observation error covariance matrix
    Q = np.array([[625, 0],
                  [0, 36]])
    sqdif_filtered = 0
    sqdif_observed = 0
    NUM_EXPERIMENTS = 1000
    for _ in range(NUM_EXPERIMENTS):
        # Generate true state based on initial state and state transition error
        for i in range(N - 1):
            x[i + 1] = A.dot(x[i]) + B.T.dot(u[i]) + rng.multivariate_normal(mean=[0, 0], cov=R)

        # Generate measurements (observed state)
        z = x + rng.multivariate_normal(mean=[0, 0], cov=Q, size=N)

        kf = LinKF(A=A, B=B, C=np.identity(2), R=R, Q=Q)
        covs = np.array([[[100, 0], [0, 100]]] * N)
        m, c = kf.run(z, covs, u, z, N)

        sqdif_observed += np.mean((x - z) ** 2, axis=0)
        sqdif_filtered += np.mean((x - m) ** 2, axis=0)

    print("Average squared error of observations: {0} over {1} experiments".format(sqdif_observed / NUM_EXPERIMENTS,
                                                                                   NUM_EXPERIMENTS))
    print("Average squared error of filter-inferred vals: {0} over {1} experiments".format(
        sqdif_filtered / NUM_EXPERIMENTS,
        NUM_EXPERIMENTS))

    plt.figure(1)
    plt.subplot(211)
    plt.plot(np.arange(N), x.T[0], marker='o', label='real x')
    plt.plot(np.arange(N), m.T[0], marker='x', linestyle='dashed', label='filtered x')
    plt.plot(np.arange(N), z.T[0], marker='+', linestyle='dashed', label='observed x')
    plt.legend(title='Legend:')
    plt.subplot(212)
    plt.plot(np.arange(N), x.T[1], marker='o', label='real v')
    plt.plot(np.arange(N), m.T[1], marker='x', linestyle='dashed', label='filtered v')
    plt.plot(np.arange(N), z.T[1], marker='+', linestyle='dashed', label='observed v')
    plt.legend(title='Legend:')
    plt.show()
