import numpy as np
from numpy.matlib import randn
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, A, B, C, R, Q):
        self.A = A  # state matrix
        self.B = B  # control matrix
        self.C = C  # measurement matrix
        self.R = R  # state transition uncertainty covariance matrix
        self.Q = Q  # measurement error covariance matrix

    def predict(self, m_prev, S_prev, u):
        m_prior = self.A.dot(m_prev) + self.B.dot(u)
        S_prior = np.dot(np.dot(self.A, S_prev), self.A.T) + self.R
        return m_prior, S_prior

    def update(self, m_prior, S_prior, z):
        kalman_gain = (S_prior.dot(self.C.T)).dot(np.linalg.inv((self.C.dot(S_prior)).dot(self.C.T) + self.Q))
        m_post = m_prior - kalman_gain.dot((z - self.C.dot(m_prior).T).T)
        S_post = (np.identity(self.A.shape[0]) - kalman_gain.dot(self.C)).dot(S_prior)
        return m_post, S_post

    def single_iter_linear(self, m_prev, S_prev, u, z):
        m_prior, S_prior = self.predict(m_prev, S_prev, u)
        m_post, S_post = self.update(m_prior, S_prior, z)
        return m_post, S_post

    def run_linear(self, mean_0, cov_0, u, z, num_iterations):
        state_means = np.ndarray((num_iterations, mean_0.shape()))
        state_covs = np.ndarray((num_iterations, cov_0.shape()))
        state_means[0] = mean_0
        state_covs[0] = cov_0
        for i in range(0, num_iterations):
            state_means[i], state_covs[i] = self.single_iter_linear(m_prev=state_means[i - 1], S_prev=state_covs[i - 1],
                                                                    u=u[i], z=z[i])
        return state_means, state_covs

    def run_linear_arr(self, bel_means, bel_covs, controls, observations, num_iterations):
        posterior_means = np.ndarray(bel_means.shape)
        posterior_means[0] = bel_means[0]
        posterior_covs = np.ndarray(bel_covs.shape)
        posterior_covs[0] = bel_covs[0]
        for i in range(1, num_iterations):
            res = self.single_iter_linear(m_prev=np.transpose([posterior_means[i - 1]]), S_prev=posterior_covs[i - 1], u=controls[i],
                                          z=observations[i])
            posterior_means[i] = res[0].reshape(2)
            posterior_covs[i] = res[1].reshape((2, 2))
        return posterior_means, posterior_covs


if __name__ == '__main__':
    x_observations = np.array([4000, 4260, 4550, 4860, 5110])
    v_observations = np.array([280, 282, 285, 286, 290])
    n = np.shape(x_observations)[0]
    z = np.stack((x_observations, v_observations)).T
    # Process / Estimation Errors
    error_est_x = 20
    error_est_v = 5
    # Observation Errors
    error_obs_x = 25
    error_obs_v = 6
    Q = np.array([[625, 0],
                  [0, 36]])
    # Initial Conditions
    a = 2  # Acceleration
    t = 1  # Difference in time
    A = np.array([[1, t], [0, 1]])
    B = np.array([[0.5 * t ** 2], [t]])
    u = np.array([a] * n)
    R = np.array([[1, 0], [0, 1]])
    kf = KalmanFilter(A=A, B=B, C=np.identity(2), R=R, Q=Q)
    covs = np.array([[[400, 0], [0, 25]]] * n)
    m, c = kf.run_linear_arr(z, covs, u, z, n)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(np.arange(n), x_observations,  marker='+', linestyle='dashed', label='observed (noisy) x')
    plt.plot(np.arange(n),m.T[0], marker='o', label='filtered x')
    plt.legend(title='Legend:')
    plt.subplot(212)
    plt.plot(np.arange(n), v_observations,  marker='+', linestyle='dashed', label='observed (noisy) v')
    plt.plot(np.arange(n),m.T[1], marker='o', label='filtered v')
    plt.legend(title='Legend:')
    plt.show()