import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# pdfs are the derivatives of cdfs
# evaluating a pdf of x does not give the probability but gives the probability density
sample = np.linspace(-3, 3, 101)
density = stats.norm.pdf(sample, 0, 1)


mean = 163
std = 52.8

plt.figure(1)
sample2 = np.linspace(mean-3*std, mean+3*std, 101)
density2 = stats.norm.pdf(sample2, mean, std)
plt.plot(density2)


# kernel density estimation - takes a sample and finds a pdf that fits the data
sample2 = np.random.normal(mean, std, 5000)
kde = stats.gaussian_kde(sample2)
plt.figure(2)
plt.plot(kde.evaluate(np.arange(130, 190)))
plt.show()

"""
PMFs, represent the probabilities for a discrete set of
values. To get from a PMF to a CDF, you add up the probability masses to
get cumulative probabilities. To get from a CDF back to a PMF, you compute
differences in cumulative probabilities. 

A PDF is the derivative of a continuous CDF; or, equivalently, a CDF is the
integral of a PDF. Remember that a PDF maps from values to probability
densities; to get a probability, you have to integrate.

To get from a discrete to a continuous distribution, you can perform various
kinds of smoothing. One form of smoothing is to assume that the data come 
from an analytic continuous distribution (like exponential or normal) and
to estimate the parameters of that distribution. Another option is kernel
density estimation.
"""