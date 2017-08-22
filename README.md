# Overview

Unsupervised clustering of sequences of arbitrary length using mixture of discrete-state markov models.

For more informations about the model, see Chap 23 pp
Barber, David. *Bayesian reasoning and machine learning*. Cambridge University Press, 2012.
APA (available: http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwik)


# Example of code

``` python
import pymmm

observations = []
observations.append([0, 2 ,2, 2, 4])
observations.append([0, 1, 2 ,4, 4, 4, 1, 1])
observations.append([0, 3, 2 ,2, 2, 1, 1, 1])
observations.append([0, 1, 2 ,2, 2, 1, 1, 1])
observations.append([0, 2, 2 ,2, 2, 1, 1, 1])
observations.append([0, 1, 2 ,2, 2, 1, 1, 1])
observations.append([0, 3, 2 ,2, 2, 4, 0, 1])
observations.append([0, 1, 2 ,1, 2, 0, 0, 1])

model = pymmm.mmm(n_cluster=2)
model.fit(observations)


obs = []
obs.append([0, 3, 2 ,0, 0, 1, 1, 1])
obs.append([0, 3, 2 ,2, 2, 1, 1, 1])
obs.append([0, 1, 1 ,1, 2, 2, 1, 1])
z = model.predict(obs)
```