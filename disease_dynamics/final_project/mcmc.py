from typing import Any
import numpy as np
from scipy.special import logit
from scipy.special import expit
from numpy.random import default_rng
from scipy.stats import binom

from constants import (
    DEFAULT_POPULATION,
    DEFAULT_NUM_TIME_STEPS,
    DEFAULT_NUM_PARTICLES,
)

rng = default_rng(seed=42)


# DEFAULT_NUM_TIME_STEPS = 12 * 5

proposal_standard_dev = np.array([0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2])
num_params = len(proposal_standard_dev)

def run_one_timestep(
    current_latent_states: np.ndarray,
    which_month: int,
    params: np.ndarray
) -> np.ndarray:
    
    beta = np.exp(params[0])
    gamma = np.exp(params[1])
    season = expit(params[5])
    peak = 12 * expit(params[6])

    sir_pop_by_particle = np.full((DEFAULT_NUM_PARTICLES, 2, 4), np.nan)

    # Initialize current state for each particle
    sir_pop_by_particle[:, 0, 0] = current_latent_states[:, 0]  # S
    sir_pop_by_particle[:, 0, 1] = current_latent_states[:, 1]  # I
    sir_pop_by_particle[:, 0, 2] = current_latent_states[:, 2]  # R

    # Compute seasonal transmission probability
    seasonal_multiplier = 1 + season * np.sin(2 * np.pi * (which_month + peak) / 12)
    infection_rate = beta * seasonal_multiplier * sir_pop_by_particle[:, 0, 1] / DEFAULT_POPULATION
    infection_prob = 1 - np.exp(-infection_rate)

    # Draw infections and recoveries
    s_to_i = rng.binomial(
        n=sir_pop_by_particle[:, 0, 0].astype(int),
        p=infection_prob
    )
    
    recovery_prob = 1 - np.exp(-gamma)
    i_to_r = rng.binomial(
        n=sir_pop_by_particle[:, 0, 1].astype(int),
        p=recovery_prob
    )

    # Update states for next timestep
    sir_pop_by_particle[:, 1, 0] = sir_pop_by_particle[:, 0, 0] - s_to_i  # Susceptible
    sir_pop_by_particle[:, 1, 1] = sir_pop_by_particle[:, 0, 1] + s_to_i - i_to_r  # infectious
    sir_pop_by_particle[:, 1, 2] = sir_pop_by_particle[:, 0, 2] + i_to_r  # Recovered
    sir_pop_by_particle[:, 1, 3] = s_to_i  # new infections

    # next timestep
    return sir_pop_by_particle[:, 1, :] 


def run_smc(
    sir_out_last_month: np.ndarray,
    current_month: int,
    params: np.ndarray,
    observed_data: np.ndarray
) -> dict[str, np.ndarray]:
    
    rho = expit(params[4])
    
    # Run one timestep of the latent SIR model
    tmp_latent_states = run_one_timestep(
        current_latent_states=sir_out_last_month,
        which_month=current_month,
        params=params
    )
    predicted_infections = tmp_latent_states[:, 3].astype(int)  # new infections
    observed_cases = int(observed_data[current_month])
    
    # Compute likelihood of observed cases given predicted infections
    tmp_likelihood = np.array([
        binom.pmf(observed_cases, pred, rho) if pred >= observed_cases else 0.0
        for pred in predicted_infections
    ])
    
    # If all likelihoods are zero, reset to a small constant
    if np.max(tmp_likelihood) == 0:
        tmp_likelihood = np.full(DEFAULT_NUM_PARTICLES, 1e-10)

    # Normalize to form a probability distribution
    weights = tmp_likelihood / np.sum(tmp_likelihood)

    # Resample particles with replacement according to weights
    sampled_indices = rng.choice(DEFAULT_NUM_PARTICLES, size=DEFAULT_NUM_PARTICLES, replace=True, p=weights)
    sir_out_this_month = tmp_latent_states[sampled_indices, :]
    
    # Package result
    out = {
        "new_latent_states": sir_out_this_month,
        "new_log_likelihoods": tmp_likelihood[sampled_indices],
        "sampled_particles": sampled_indices
    }
    return out


def initialize_array(params: np.ndarray) -> np.ndarray:
    s_init_frac = expit(params[2])
    i_init_frac = expit(params[3]) * (1 - s_init_frac)

    s_init = round(s_init_frac * DEFAULT_POPULATION)
    i_init = round(i_init_frac * DEFAULT_POPULATION)
    r_init = DEFAULT_POPULATION - (s_init + i_init)

    sir_out_all_months = np.full((DEFAULT_NUM_PARTICLES, DEFAULT_NUM_TIME_STEPS + 1, 4), np.nan)

    # Set initial state
    sir_out_all_months[:, 0, 0] = s_init  # Susceptible
    sir_out_all_months[:, 0, 1] = i_init  # Infected
    sir_out_all_months[:, 0, 2] = r_init  # Recovered
    sir_out_all_months[:, 0, 3] = 0    # New infections

    return sir_out_all_months


def calc_log_likelihood(
    params: np.ndarray,
    observed_data: np.ndarray
) -> float:
    sir_out_all_months = initialize_array(params)
    val = 0.0

    for month_step in range(1, DEFAULT_NUM_TIME_STEPS + 1):
        tmp_out = run_smc(
            sir_out_last_month=sir_out_all_months[:, month_step - 1, :],
            current_month=month_step - 1,
            params=params, 
            observed_data=observed_data
        )

        # Resample all history up to this month based on sampled particles
        for t in range(month_step):
            sir_out_all_months[:, t, :] = sir_out_all_months[tmp_out["sampled_particles"], t, :]

        # Save new latent states
        sir_out_all_months[:, month_step, :] = tmp_out["new_latent_states"]

        # Add log of mean likelihood
        mean_likelihood = np.mean(tmp_out["new_log_likelihoods"])
        new_val = np.log(mean_likelihood if mean_likelihood > 0 else 1e-10)  # avoid log(0)
        val += new_val

    return val, sir_out_all_months


def proposal_draw(proposal_standard_dev: np.ndarray) -> np.ndarray:
    # Using multivariate normal with mean 0 and diagonal covariance matrix
    vec = rng.multivariate_normal(mean=np.zeros(num_params), cov=np.diag(proposal_standard_dev**2))
    return vec


def propose_new_val(
    current_parameter_guess: np.ndarray,
    proposal_standard_dev: np.ndarray,
    observed_data: np.ndarray,
) -> dict[str, np.ndarray]:
    new_parameter_guess = current_parameter_guess + proposal_draw(proposal_standard_dev)
    new_chain_info = calc_log_likelihood(new_parameter_guess, observed_data)
    out = {
        "new_parameter_guess": new_parameter_guess,
        "new_chain_info": new_chain_info
    }
    return out


def metropolis_hastings_step(
    current_parameter_guess: np.ndarray,
    new_parameter_guess: np.ndarray,
    current_log_likelihood: float,
    new_log_likelihood: float,
    current_latent: np.ndarray,
    new_latent: np.ndarray,
) -> dict[str, Any]:
    alpha = np.exp(new_log_likelihood - current_log_likelihood)
    r_num = rng.uniform()  # use np.random.uniform() if not using a seeded rng

    accept_step = r_num < alpha

    if accept_step:
        next_parameter_guess = new_parameter_guess
        next_log_likelihood = new_log_likelihood
        next_latent = new_latent

    else:
        next_parameter_guess = current_parameter_guess
        next_log_likelihood = current_log_likelihood
        next_latent = current_latent

    out = {
        "alpha": alpha,
        "r_num": r_num,
        "accept_step": accept_step,
        "next_parameter_guess": next_parameter_guess,
        "next_latent": next_latent,
        "next_log_likelihood": next_log_likelihood,
        "new_parameter_guess": new_parameter_guess,
        "new_log_likelihood": new_log_likelihood
    }
    return out


def mcmc_step(all_steps: dict, proposal_standard_dev: np.ndarray, observed_data: np.ndarray) -> dict:
    chain = all_steps["chain"]
    latents = all_steps["latents"]
    accept_chain = all_steps["accept_chain"]
    log_likelihood = all_steps["log_likelihood"]
    
    current_parameter_guess = chain[:, -1]
    current_latent = latents[-1]
    current_log_likelihood = log_likelihood[-1]
    
    new_param_info = propose_new_val(
        current_parameter_guess=current_parameter_guess,
        proposal_standard_dev=proposal_standard_dev,
        observed_data=observed_data
    )
    
    new_parameter_guess = new_param_info["new_parameter_guess"]
    new_log_likelihood = new_param_info["new_chain_info"][0]
    new_latent = new_param_info["new_chain_info"][1]
    
    metropolis_hastings_info = metropolis_hastings_step(
        current_parameter_guess=current_parameter_guess,
        new_parameter_guess=new_parameter_guess,
        current_log_likelihood=current_log_likelihood,
        new_log_likelihood=new_log_likelihood,
        current_latent=current_latent,
        new_latent=new_latent,
    )
    
    accept_chain.append(int(metropolis_hastings_info["accept_step"]))
    chain = np.column_stack((chain, metropolis_hastings_info["next_parameter_guess"]))
    latents.append(metropolis_hastings_info["next_latent"])
    log_likelihood.append(metropolis_hastings_info["next_log_likelihood"])
    
    out = {
        "chain": chain,
        "latents": latents,
        "accept_chain": accept_chain,
        "log_likelihood": log_likelihood,
        "current_parameter_guess": current_parameter_guess,
        "new_parameter_guess": new_parameter_guess,
        "MH_info": metropolis_hastings_info
    }
    return out


def mcmc_steps(
    n_steps: int,
    all_steps: dict,
    proposal_standard_dev: np.ndarray,
    observed_data: np.ndarray
) -> dict:
    for _ in range(n_steps):
        all_steps = mcmc_step(all_steps, proposal_standard_dev, observed_data)
    return all_steps
