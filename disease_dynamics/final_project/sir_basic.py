"""SIR model simulation with basic structure and seasonality."""
import numpy as np
from scipy.special import expit
from numpy.random import default_rng

from constants import (
    DEFAULT_POPULATION,
    DEFAULT_NUM_TIME_STEPS,
)

rng = default_rng(seed=42)


def sir_out(params: np.ndarray, population: int = DEFAULT_POPULATION) -> np.ndarray:
    """
    Simulate the SIR model with given parameters.

    :param params: Parameters for the SIR model in the following order:
        - beta: Transmission rate (log scale)
        - gamma: Recovery rate (log scale)
        - s_init_frac: Initial susceptible fraction (logit scale)
        - i_init_frac: Initial infected fraction (logit scale)
        - rho: Reporting rate (logit scale)
        - season: Seasonality factor (logit scale)
        - peak: Peak month for seasonality (logit scale)
    :param population: Total population size.

    :return: SIR population array with columns:
        - S: Susceptible individuals
        - I: Infected individuals
        - R: Recovered individuals
        - NewI: New infections in the current step
        - NewR: New recoveries in the current step
        - ObsCases: Observed cases in the current step
        - InfectionRate: Infection rate for the current step
    """
    beta = np.exp(params[0])
    gamma = np.exp(params[1])
    s_init_frac = expit(params[2])
    i_init_frac = expit(params[3]) * (1 - s_init_frac)

    rho = expit(params[4])
    season = expit(params[5])
    peak = 12 * expit(params[6])

    s_init = round(s_init_frac * population)
    i_init = round(i_init_frac * population)
    r_init = population - (s_init + i_init)

    # Columns: S, I, R, NewI, NewR, ObsCases, InfectionRate
    sir_population = np.full((DEFAULT_NUM_TIME_STEPS + 1, 7), np.nan)

    sir_population[0, 0] = s_init
    sir_population[0, 1] = i_init
    sir_population[0, 2] = r_init

    for r_num in range(DEFAULT_NUM_TIME_STEPS):
        tmp_s = sir_population[r_num, 0]
        tmp_i = sir_population[r_num, 1]
        tmp_r = sir_population[r_num, 2]

        tmp_seasonal_beta = beta * (1 + season * np.sin(2 * np.pi * (r_num + peak) / 12))

        infection_prob = 1 - np.exp(-tmp_seasonal_beta * tmp_i / population)
        recovery_prob = 1 - np.exp(-gamma)
        sir_population[r_num, 6] = infection_prob

        s_to_i = rng.binomial(int(tmp_s), infection_prob)
        i_to_r = rng.binomial(int(tmp_i), recovery_prob)

        sir_population[r_num + 1, 0] = tmp_s - s_to_i
        sir_population[r_num + 1, 1] = tmp_i + s_to_i - i_to_r
        sir_population[r_num + 1, 2] = tmp_r + i_to_r

        sir_population[r_num, 3] = s_to_i
        sir_population[r_num, 4] = i_to_r
        
        observed_cases = rng.binomial(s_to_i, rho)
        sir_population[r_num, 5] = observed_cases

    return sir_population
