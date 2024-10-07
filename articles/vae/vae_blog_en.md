# Variationnal Auto Encoder (VAE)
In recent articles about image generation, the state-of-art seems to be accomplished by a model named *Variationnal Auto Encoder*.
To understand this model we need to go further in two keys concept : *Variationnal Inference* (VI) and *Auto Encoder* (AE).

## Varitional Inference (VI)
Variationnal Inference is a branch of statistic which try to approach an unkonwn distribution from known distributions $\{q_{\theta}\}$. A way to approach the unknown distribution is for example by adjusting parameters of the knowned distrbution.

For example if you want to approach a given distribtuion by using a Gaussian familly. You have to set correctly $\mu$ and $\sigma$ until your gaussian distribution $q_{\mu,\sigma}$ is *similar* to the distibution objective.
![gif](image/animation_pillow.gif)

To start a *Variational Inference* it is necessary to get : 
1. A familly of parameterized known distributions $\{q_{\theta}\}$.
2. A metric to compare how far we are of the unkown distibution.

A natural metric to compare the divergence between two distibution $p$ and $q$ is the **Kullback-Leiber divergence** KL : 
$\textbf{KL(p||q)}=\int_Z q(z).\log{\frac{p(z)}{q(z)}}$ where Z is the definition set of p and q.

### General formulation of VI problem
In pratice, we want to estimate a posterior distribution in order to solve machine learning problem.
Let $p(z|x)$ a posterior distribution and $p(x|z)$ the associates likelihood.

As a consequence, a *Varitional Inference* (VI) problem with KL-divergence as a metric is formulated as :
$$ \argmin_{q\in Q} \textbf{KL}(q||p(z|x))$$
With expectation manipulations and Bayes rule the $\textbf{KL}$ become :

$\textbf{KL}(q||p(z|x))=-$<span style="border: 2px solid red; padding:3px">$(E[\log{q(z)}]-E[\log{p(x|z)}.p(z)])$</span>+$\log{p(x)}$

Maximise the red box is equivalent to minimize the $\textbf{KL}$.
This quantity in the red box is the **ELBO** (Evidence Lower Bound).

### How to solve a VI problem ? What are known quatities ?
-> CAVI
-> Mean-field Approximation
### Stochastic VI

## Auto-Encoder

## Variationnal Auto-Encoder (VAE)

### Application in image generation
### Nouveau Variationnal Auto-Encoder (NVAE)