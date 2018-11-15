#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import matplotlib.pyplot as plt


class mmm():
    def __init__(self, n_cluster=2,
                 transition_probs=[], prior=[], pi=[]):
        self.n_cluster = n_cluster
        self.transition_probs = transition_probs
        self.prior = prior
        self.pi = pi

        self.observations = []
        self.n_observations = []
        self.dim_state_space = 0
        self.posterior = []

        self.likelihood_bic = 0
        self.likelihood_aic = 0

    def predict(self, observations):
        # likelihood of new observations
        log_pi = log_inf(self.pi)
        log_alpha = log_inf(self.prior)
        log_A = log_inf(self.transition_probs)
        n_obs = len(observations)
        # trans_matrix = map(lambda x: compute_transition_matrix(
        #     x, self.dim_state_space), observations)
        trans_matrix = [compute_transition_matrix(x, self.dim_state_space)
                        for x in observations]
        trans_matrix = np.array(trans_matrix).transpose()
        trans_matrix = np.reshape(
            trans_matrix, (self.dim_state_space*self.dim_state_space, n_obs))

        log_A = np.reshape(log_A, (self.dim_state_space *
                                   self.dim_state_space, self.n_cluster))
        x1 = np.zeros((self.dim_state_space, n_obs))
        for s in range(n_obs):
            x1[observations[s][0], s] = 1

        llikehood_iter = np.dot(log_pi.transpose(), x1) + \
            np.dot(log_A.transpose(), trans_matrix) + \
            np.dot(log_alpha, np.ones((1, n_obs)))
        posterior = normalize_exp(llikehood_iter)
        return posterior

    def fit(self, observations=[], max_iter=1000, threshold=1e-8, verbose=False):
        # Inputs:
        #   observations        : list of sequences
        #   dim_state_space     : dimension of the state space (number of possible actions)
        #   n_cluster           : number of mixtures
        #   max_iter            : maximum EM iterations
        #   theshold            : criteria to stop EM if the likelihood increase is less that the threshold

        # Outputs:
        #   alpha        : prior distribution of the mixtures
        #   pi           : conditional distribution of the initial states
        #   A            : conditional distribution of state transition p(x_{s,n}|x_{s,n-1}, z_n)
        #   z            : posterior distribution of the mixture p(z|\Xbf, pi, rho, A) [or posterior of the latent variables]
        #   likelihood   : likelihood of each iterations

        self.n_observations = len(observations)  # number of sequences
        self.dim_state_space = len(
            np.unique(list(itertools.chain(*observations))))

        # Computation of the transition matrix
        # trans_matrix = map(lambda x: compute_transition_matrix(
        #     x, self.dim_state_space), observations)
        trans_matrix = [compute_transition_matrix(x, self.dim_state_space)
                        for x in observations]
        trans_matrix = np.array(trans_matrix).transpose()
        trans_matrix = np.reshape(
            trans_matrix, (self.dim_state_space*self.dim_state_space, self.n_observations))

        # Initial state
        x1 = np.zeros((self.dim_state_space, self.n_observations))
        for s in range(self.n_observations):
            x1[observations[s][0], s] = 1

        # Random initialization
        self.prior = normalize(np.random.rand(self.n_cluster, 1))
        self.pi = normalize(np.random.rand(
            self.dim_state_space, self.n_cluster))
        self.transition_probs = normalize(np.random.rand(
            self.dim_state_space, self.dim_state_space, self.n_cluster))
        self.posterior = np.zeros((self.n_cluster, self.n_observations))

        # Initialization of the likelihood vector
        likelihood = np.zeros(max_iter)
        diff_likelihood = 100

        for iter in range(max_iter):

            # Parameters to compute the log likelihood
            log_pi = log_inf(self.pi)
            log_alpha = log_inf(self.prior)
            log_A = log_inf(self.transition_probs)
            log_A = np.reshape(
                log_A, (self.dim_state_space*self.dim_state_space, self.n_cluster))

            # E-step:
            # llikehood_iter: matrix of log likelihood
            # llikehood_iter = log(
            # |  p(z_1=1)p_\theta(x_S|z_S=1)   ...                  p(z_S=1)p_\theta(x_S|z_S=1)  |
            # |  .                                                                               |
            # |  .                                                                               |
            # |  p(z_1=K)p_\theta(x_1|z_1=K)                        p(z_S=K)p_\theta(x_S|z_S=K)  |
            # )

            # Log likelihood of this iteration
            llikehood_iter = np.dot(log_pi.transpose(), x1) + \
                np.dot(log_A.transpose(), trans_matrix) + \
                np.dot(log_alpha, np.ones((1, self.n_observations)))

            # NAN
            if (np.isnan(llikehood_iter)).any():
                print('likelihood na')
                break

            self.posterior = normalize_exp(llikehood_iter)

            # M-step:
            # dim= dim_state_spacexn_cluster
            self.pi = normalize(np.dot(x1, self.posterior.transpose()))
            self.transition_probs = normalize(np.reshape(np.dot(trans_matrix, self.posterior.transpose()),
                                                         (self.dim_state_space, self.dim_state_space, self.n_cluster)))

            if np.isnan(self.transition_probs).any():
                print('A na')
                break
            self.prior = normalize(
                np.sum(self.posterior, 1).reshape((self.n_cluster, 1)))

            likelihood_tmp = np.exp(llikehood_iter).sum(0)
            likelihood_tmp = log_inf(likelihood_tmp).sum()
            if iter > 0:
                diff_likelihood = likelihood_tmp - likelihood[iter-1]
            likelihood[iter] = likelihood_tmp
            if np.abs(diff_likelihood) < threshold:
                likelihood = likelihood[0:iter]
                break

        # Computation of the sample size and degree of freedom for BIC/AIC:
        # https://en.wikipedia.org/wiki/Akaike_information_criterion
        sample_size = np.sum([len(x) for x in observations])
        # sample_size = np.sum(np.array(map(lambda x: len(x), observations)))
        n_free = np.prod(self.pi.shape) + \
            np.prod(self.transition_probs.shape) + \
            np.prod(self.prior.shape) - \
            self.transition_probs.shape[0] - self.pi.shape[0] - 1
        # n_free = parameters to estimate - normalization
        self.likelihood_bic = -2*(likelihood[-1]) + n_free*np.log(sample_size)
        self.likelihood_aic = 2*n_free - 2*(likelihood[-1])

        return True
        # return alpha, pi, A, likelihood, likelihood_bic, likelihood_aic, z

    def plot_transition_matrix(self, cluster=0, title="",
                               ticks=['home', 'mediation', 'search',
                                      'download', 'consult']):

        # Plot function for datavisualization
        data = self.transition_probs[:, :, cluster].transpose()
        prior = self.pi[:, cluster].transpose()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(data, interpolation='nearest',
                         vmin=0, vmax=1, cmap='Reds')
        fig.colorbar(cax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # ax2 = fig.add_subplot(212)
        # cax2 = ax2.matshow(np.asarray([prior]*2), interpolation='nearest',vmin=0, vmax=1, cmap='Reds')
        plt.title(title)

        ax.set_xticklabels(['']+ticks)
        ax.set_yticklabels(['']+ticks)
        fontsize = 15
        params = {'axes.labelsize': fontsize + 2,
                  'font.size': fontsize,
                  'legend.fontsize': fontsize,
                  'xtick.labelsize': fontsize,
                  'ytick.labelsize': fontsize}
        plt.rcParams.update(params)
        plt.tight_layout()
        return plt

        # fig, axes = plt.subplots(nrows=2, ncols=1)
        # cax = axes[0].matshow(data, interpolation='nearest',vmin=0, vmax=1, cmap='Reds')
        # cax2 = axes[1].matshow(np.asarray([prior]*2), interpolation='nearest',vmin=0, vmax=1, cmap='Reds')

        # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # fig.colorbar(cax2, cax=cbar_ax)


def compute_transition_matrix(x, n, norm=0):
    # Compute the transition matrix from data (see
    # http://stackoverflow.com/questions/13219041/how-can-i-speed-up-transition-matrix-creation-in-numpy)
    flat_coords = np.ravel_multi_index((x[:-1], x[1:]), (n, n))
    trans_matrix = np.bincount(flat_coords, minlength=n*n).reshape((n, n))
    return trans_matrix


def log_inf(X, replace_inf=-500):
    # function that deals with log(0)
    log_x = np.log(X)
    log_x[np.isinf(log_x)] = replace_inf
    return log_x


def normalize_exp(X):
    # Normalize each columns of X
    # Numerically stable
    X_max = np.max(X, 0)
    X_size = X.shape
    X = X-np.tile(X_max, (X_size[0], 1))
    return normalize(np.exp(X))


def normalize(X):
    # Normalize each columns of X
    # TODO: could be way better
    X_size = X.shape
    col_sums = X.sum(axis=0)
    col_sums = col_sums  # + (col_sums==0)
    X_norm = np.zeros(X_size)
    X_norm = np.nan_to_num(X.astype(np.float64) / col_sums[np.newaxis, :])
    return X_norm

    # elif len(X_size)==3:
    #     for ndim in range(X_size[-1]):
    #         col_sum = col_sums[:,ndim]
    #         X_norm[:,:,ndim] = X[:,:,ndim].astype(np.float64) / col_sum[np.newaxis, :]
    # return X_norm


def get_list_actions_v2(df_list_sessions, possible_actions=['homepage', 'mediation', 'sru', 'download', 'navigation']):
    # Transform a binary n dimension dataframe into categorial vector
    list_actions = []
    nactions = len(possible_actions)
    idx = -1
    for session in df_list_sessions:
        actions = []
        session = session[possible_actions]
        cols = session.columns
        dim_resize = session.shape[0]*session.shape[1]
        action = np.where(np.array(session).reshape(dim_resize, 1))[
            0] % nactions
        map(lambda x: actions.append(x), action)
        list_actions.append(actions)
    return list_actions
