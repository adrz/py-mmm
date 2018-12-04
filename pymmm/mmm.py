#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import matplotlib.pyplot as plt


class MixtureMarkovChains():
    """ Expectation Maximization for mixture of discret markov models.

    Unsupervised sequences clustering of arbitrary length.
    """
    def __init__(self, n_cluster: int=2,
                 transition_probs: list=[],
                 prior: list=[], pi: list=[]):
        self.n_cluster = n_cluster
        self.transition_probs = transition_probs
        self.prior = prior
        self.pi = pi

        self.n_observations = None
        self.dim_state_space = None
        self.posterior = []

        self.likelihood_bic = None
        self.likelihood_aic = None

    def predict(self, observations: list):
        """ Return the posterior probability of a each markov models
        given the sequences ''observations''.
        Previous usage fit is required.

        Parameters
        ----------
        observations (list): list of sequences

        Returns
        ----------
        Posterior. In order to get the clusters use:
        np.argmax(posterior, 0)
        """

        # log initial states
        log_pi = log_inf(self.pi)

        # log prior
        log_alpha = log_inf(self.prior)

        # log transition matrix
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

        # compute log posterior
        log_posterior = (np.dot(log_pi.transpose(), x1) +
                         np.dot(log_A.transpose(), trans_matrix) +
                         np.dot(log_alpha, np.ones((1, n_obs))))
        posterior = normalize_exp(log_posterior)
        return posterior

    def fit(self, observations: list, max_iter: int=1000,
            threshold: float=1e-8, verbose=False):
        """ Expectation maximization

        Compute:
        alpha        : prior distribution of the mixtures
        pi           : conditional distribution of the initial states
        A            : conditional distribution of state transition p(x_{s,n}|x_{s,n-1}, z_n)
        z            : posterior distribution of the mixture p(z|\Xbf, pi, rho, A) 
        (or posterior of the latent variables]
        likelihood   : likelihood of each iterations
 

        Parameters
        ----------
        observations (list): list of sequences allow
        dim_state_space (int): dimension of the state space (number of possible actions)
        max_iter (int): maximum number of EM iterations.
        threshold (float): criteria to stop EM if the likelihood no longer
        increase more than this value.

        Returns:
        """

        # determine the number of sequences and the state space
        self.n_observations = len(observations)
        self.dim_state_space = len(
            np.unique(list(itertools.chain(*observations))))

        # compute the transition matrix and reshape it to allow a
        # faster computation of the likelihood with vectorization trick.
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

            # Check if numerical issue (which is frequent)
            if (np.isnan(llikehood_iter)).any():
                print('Likelihood numerical issue')
                break

            self.posterior = normalize_exp(llikehood_iter)

            # M-step:
            # dim= dim_state_spacexn_cluster
            self.pi = normalize(np.dot(x1, self.posterior.transpose()))
            self.transition_probs = normalize(
                np.reshape(np.dot(trans_matrix,
                                  self.posterior.transpose()),
                           (self.dim_state_space, self.dim_state_space, self.n_cluster))
            )

            if np.isnan(self.transition_probs).any():
                print('Transition probability numerical issue')

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

        # degree of freedom
        n_free = (np.prod(self.pi.shape) +
                  np.prod(self.transition_probs.shape) +
                  np.prod(self.prior.shape) -
                  self.transition_probs.shape[0] - self.pi.shape[0] - 1)

        self.likelihood_bic = -2*(likelihood[-1]) + n_free*np.log(sample_size)
        self.likelihood_aic = 2*n_free - 2*(likelihood[-1])

    def plot_transition_matrix(self, cluster: int=0, title: str="Transition Matrix",
                               ticks=['home', 'mediation', 'search',
                                      'download', 'consult']):
        """ Representation of the transition matrix of a cluster

        Parameters
        ----------
        cluster (int): id of the cluster to be represented (should be 0 =< cluster < n_clusters)
        title (str): title of the plot
        ticks (str): string associate to the elements of a sequence.
        A sequence is represented by a succession of integer but often has a signification.
        
        Return
        ----------
        matplotlib.plot
        """

        # Plot function for datavisualization
        data = self.transition_probs[:, :, cluster].transpose()
        prior = self.pi[:, cluster].transpose()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(data, interpolation='nearest',
                         vmin=0, vmax=1, cmap='Reds')
        fig.colorbar(cax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
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


def compute_transition_matrix(x: list, n: int):
    """ Fast computation of transition matrix for one sequence
    see: http://stackoverflow.com/questions/13219041/how-can-i-speed-up-transition-matrix-creation-in-numpy)

    Parameters
    ----------
    x (list): sequence
    n (int): number of states

    Returns:
    ----------
    np.array
    """

    # Compute the transition matrix from data (see
    flat_coords = np.ravel_multi_index((x[:-1], x[1:]), (n, n))
    trans_matrix = np.bincount(flat_coords, minlength=n*n).reshape((n, n))
    return trans_matrix


def log_inf(x, replace_inf=-500):
    """ Numerically safe computation of logarithm
    Bound the logarithm to replace_inf. Allow us to deals with log(0)
    """
    log_x = np.log(x)
    log_x[np.isinf(log_x)] = replace_inf
    return log_x


def normalize_exp(x):
    """ Numerically stable 'l1' normalization of the columns of X """
    x_max = np.max(x, 0)
    x_size = x.shape
    x = x-np.tile(x_max, (x_size[0], 1))
    return normalize(np.exp(x))


def normalize(x):
    """ 'l1' normalization  """
    x_size = x.shape
    col_sums = x.sum(axis=0)
    col_sums = col_sums
    x_norm = np.zeros(x_size)
    x_norm = np.nan_to_num(x.astype(np.float64) / col_sums[np.newaxis, :])
    return x_norm


def get_list_actions(df_list_sessions,
                     possible_actions=['homepage', 'mediation', 'sru', 'download', 'navigation']):
    # Transform a binary n dimension dataframe into categorial vector
    list_actions = []
    nactions = len(possible_actions)
    idx = -1
    for session in df_list_sessions:
        actions = []
        session = session[possible_actions]
        cols = session.columns
        dim_resize = session.shape[0]*session.shape[1]
        action = np.where(
            np.array(session).reshape(dim_resize, 1))[0] % nactions

        for act in action:
            actions.append(act)

        list_actions.append(actions)
    return list_actions
