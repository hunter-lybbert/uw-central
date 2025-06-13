# Infectious Disease Dynamics Final Project

We model the spread of the measles pathogen in New York and Vermont from January 1 $^{\text{st}}$, 1930 to January 1 $^{\text{st}}$, 1940 using an SIR model.
Our model has been augmented from a basic SIR model to incorporate annual seasonality, births and deaths, and a multi-year seasonality term for the strength of the strain of the pathogen.
To optimize and fit our model, we adapted to python, the R implementation of particle MCMC provided to us.

Important files in the project:
* Data cleaning and preparation $\rightarrow$ [data_cleaning.ipynb](data_cleaning.ipynb)
* Model Fitting $\rightarrow$ [mcmc_seasonality.ipynb](mcmc_seasonality.ipynb)
* pMCMC implimentation $\rightarrow$ [mcmc/](mcmc/)