# Overview

Unsupervised clustering of sequences of arbitrary length using mixture of discrete-state markov models.

For more informations about the model, see Chap 23 pp
Barber, David. *Bayesian reasoning and machine learning*. Cambridge University Press, 2012.
APA (available: http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwik)

# Installation

```bash
$ git clone git@github.com:adrz/py-mmm.git
$ cd py-mmm
$ virtualenv -p python3 env
$ source env/bin/activate
$ pip install -r requirements.txt
```

# Example of code

Clustering a mixted length sequences.
The number of states is 5, and the number of wished clusters is 3:

``` python
from pymmm.mmm import MixtureMarkovChains
import numpy as np

model = MixtureMarkovChains(n_cluster=3)

observations = [[0, 2, 2, 2, 4],
                [0, 1, 2, 4, 4, 4, 1, 1, 1],
                [0, 3, 2, 2, 2, 1, 1, 1],
                [0, 1, 2, 2, 2, 1, 1, 1],
                [0, 2, 2, 2, 2, 2, 1, 1, 1],
                [0, 1, 2, 2, 2, 1, 1, 1],
                [0, 3, 2, 2, 2, 4, 0, 1],
                [0, 1, 2, 1, 2, 0, 0, 0, 1]]
model.fit(observations)


z_training = model.predict(observations)

observation_testing = [[0, 3, 2, 0, 0, 1, 1, 1],
                       [0, 3, 2, 2, 2, 1, 1, 1],
                       [0, 1, 1, 1, 2, 2, 1, 1]]

z_testing = model.predict(observation_testing)

# Posterior:training
print('Posterior training')
print(z_training)
print('Label training')
print(np.argmax(z_training, 0))

print('Posterior testing')
print(z_testing)
print('Label testing')
print(np.argmax(z_testing, 0))
```
