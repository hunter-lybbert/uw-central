"""
SIR model simulation with augmented structure.

Pathogen Specific Parameters:
- `beta`: Transmission rate
- `gamma`: Recovery rate
- `season`: Seasonality factor
- `peak`: Peak month for seasonality

System or Locational Parameters:
- `s_init_frac_ny`: Initial susceptible fraction in New York
- `i_init_frac_ny`: Initial infected fraction in New York
- `rho_ny`: Reporting rate in New York

- `s_init_frac_vt`: Initial susceptible fraction in Vermont
- `i_init_frac_vt`: Initial infected fraction in Vermont
- `rho_vt`: Reporting rate in Vermont

total_params = 10
"""
import numpy as np
from scipy.special import expit
from numpy.random import default_rng

from constants import (
    DEFAULT_POPULATION,
    DEFAULT_NUM_TIME_STEPS,
)

rng = default_rng(seed=42)


def setup_sir_pop(
    s_init_frac_param: float,
    i_init_frac_param: float,
    population: int = DEFAULT_POPULATION,
    default_num_time_steps: int = DEFAULT_NUM_TIME_STEPS,
) -> np.ndarray:
    """
    Initialize the SIR population array.

    :param s_init_frac_param: Initial susceptible fraction parameter.
    :param i_init_frac_param: Initial infected fraction parameter.
    :param population: Total population size.
    :param default_num_time_steps: Number of time steps for the simulation.

    :return: Initialized SIR population array.
    """
    s_init_frac = expit(s_init_frac_param)
    i_init_frac = expit(i_init_frac_param) * (1 - s_init_frac)
    s_init = round(s_init_frac * population)
    i_init = round(i_init_frac * population)
    r_init = population - (s_init + i_init)

    # Columns: S, I, R, NewI, NewR, ObsCases, InfectionRate
    sir_pop = np.full((default_num_time_steps + 1, 7), np.nan)
    sir_pop[0, 0] = s_init
    sir_pop[0, 1] = i_init
    sir_pop[0, 2] = r_init
    return sir_pop


def run_sir_step(
    sir_pop: np.ndarray,
    beta: float,
    gamma: float,
    season: float,
    peak: float,
    rho: float,
    r_num: int,
) -> np.ndarray:
    """
    Run a single step of the SIR model simulation.

    :param sir_pop: Current SIR population array.
    :param beta: Transmission rate.
    :param gamma: Recovery rate.
    :param season: Seasonality factor.
    :param peak: Peak month for seasonality.
    :param rho: Reporting rate.
    :param r_num: Current time step index.

    :return: Updated SIR population array after the step.
    """
    tmp_s = sir_pop[r_num, 0]
    tmp_i = sir_pop[r_num, 1]
    tmp_r = sir_pop[r_num, 2]
    tmp_n = tmp_s + tmp_i + tmp_r

    tmp_seasonal_beta = beta * (1 + season * np.sin(2 * np.pi * (r_num + peak) / 12))

    infection_prob = 1 - np.exp(-tmp_seasonal_beta * tmp_i / tmp_n)
    recovery_prob = 1 - np.exp(-gamma)
    sir_pop[r_num, 6] = infection_prob

    s_to_i = rng.binomial(int(tmp_s), infection_prob)
    i_to_r = rng.binomial(int(tmp_i), recovery_prob)

    sir_pop[r_num + 1, 0] = tmp_s - s_to_i
    sir_pop[r_num + 1, 1] = tmp_i + s_to_i - i_to_r
    sir_pop[r_num + 1, 2] = tmp_r + i_to_r

    sir_pop[r_num, 3] = s_to_i
    sir_pop[r_num, 4] = i_to_r
    
    observed_cases = rng.binomial(s_to_i, rho)
    sir_pop[r_num, 5] = observed_cases

    return sir_pop


def sir_out(
    params: np.ndarray,
    population_ny: int = DEFAULT_POPULATION,
    population_vt: int = DEFAULT_POPULATION,
    default_num_time_steps: int = DEFAULT_NUM_TIME_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SIR model simulation with augmented structure for New York and Vermont.

    :param params: Parameters for the SIR model.
    :param population_ny: Population size for New York.
    :param population_vt: Population size for Vermont.
    :param default_num_time_steps: Number of time steps for the simulation.

    :return: Tuple of SIR population arrays for New York and Vermont.
    """
    beta = np.exp(params[0])
    gamma = np.exp(params[1])
    season = expit(params[2])
    peak = 12 * expit(params[3])

    sir_pop_ny = setup_sir_pop(params[4], params[5], population_ny, default_num_time_steps)
    rho_ny = expit(params[6])

    sir_pop_vt = setup_sir_pop(params[7], params[8], population_vt, default_num_time_steps)
    rho_vt = expit(params[9])

    for r_num in range(default_num_time_steps):
        sir_pop_ny = run_sir_step(
            sir_pop=sir_pop_ny,
            beta=beta,
            gamma=gamma,
            season=season,
            peak=peak,
            rho=rho_ny,
            r_num=r_num,
        )
        sir_pop_vt = run_sir_step(
            sir_pop=sir_pop_vt,
            beta=beta,
            gamma=gamma,
            season=season,
            peak=peak,
            rho=rho_vt,
            r_num=r_num,
        )

    return sir_pop_ny, sir_pop_vt
