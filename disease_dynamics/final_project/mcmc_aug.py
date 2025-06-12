from typing import Any, Generator
import numpy as np
from scipy.special import expit
from scipy.stats import binom

from constants import (
    NEW_YORK_POP_1930,
    VERMONT_POP_1930,
    DEFAULT_NUM_TIME_STEPS,
    DEFAULT_NUM_PARTICLES,
    DEFAULT_NUM_US_STATES_IN_MODEL,
)


# TODO Change how the number of params is defined
# proposal_standard_dev = np.array([0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2])
# num_params = len(proposal_standard_dev)


def initialize_sir_particles(
    current_latent_states: np.ndarray,
    num_particles: int = DEFAULT_NUM_PARTICLES,
    num_us_states_in_model: int = DEFAULT_NUM_US_STATES_IN_MODEL,
) -> np.ndarray:
    """
    Initialize the SIR population for each particle.

    :param current_latent_states: Current latent states of the SIR model for each particle.
        Shape: (num_particles, 3) where columns are [S, I, R].
    :param num_particles: Number of particles to initialize.

    :return: Initialized SIR population array for each particle.
        Shape: (num_particles, 2, 4) where:
        - First dimension: Particle index
        - Second dimension: Time step (0 for current, 1 for next)
        - Third dimension: [S, I, R, NewI]
    """
    sir_pop_by_particle = np.full((num_particles, 2, 4 * num_us_states_in_model), np.nan)

    # Initialize current state for each particle
    sir_pop_by_particle[:, 0, 0] = current_latent_states[:, 0]  # NY S
    sir_pop_by_particle[:, 0, 1] = current_latent_states[:, 1]  # NY I
    sir_pop_by_particle[:, 0, 2] = current_latent_states[:, 2]  # NY R
    
    sir_pop_by_particle[:, 0, 4] = current_latent_states[:, 4]  # VT S
    sir_pop_by_particle[:, 0, 5] = current_latent_states[:, 5]  # VT I
    sir_pop_by_particle[:, 0, 6] = current_latent_states[:, 6]  # VT R

    return sir_pop_by_particle


# TODO: You need to update the run_one_timestep to work with multiple populations and states
def run_one_timestep(
    current_latent_states: np.ndarray,
    which_month: int,
    params: np.ndarray,
    rng: Generator,
    num_particles: int = DEFAULT_NUM_PARTICLES,
) -> np.ndarray:
    """
    Run one timestep of the SIR model with seasonal transmission.

    :param current_latent_states: Current latent states of the SIR model for each particle.
    :param which_month: The current month (0-11).
    :param params: Parameters for the SIR model in the following order:
        - beta: Transmission rate (log scale)
        - gamma: Recovery rate (log scale)
        - season: Seasonality factor (logit scale)
        - peak: Peak month for seasonality (logit scale)
        - s_init_frac_ny: Initial susceptible fraction in New York (logit scale)
        - i_init_frac_ny: Initial infected fraction in New York (logit scale)
        - s_init_frac_vt: Initial susceptible fraction in Vermont (logit scale)
        - i_init_frac_vt: Initial infected fraction in Vermont (logit scale)
        - rho_ny: Reporting rate in New York (logit scale)
        - rho_vt: Reporting rate in Vermont (logit scale)
    :param rng: Random number generator for reproducibility.
    :param num_particles: Number of particles to simulate.
    
    :return: Updated latent states of the SIR model for each particle.
    """
    beta = np.exp(params[0])
    gamma = np.exp(params[1])
    season = expit(params[2])
    peak = 12 * expit(params[3])

    sir_pop_by_particle = initialize_sir_particles(
        current_latent_states=current_latent_states,
        num_particles=num_particles,
    )

    tmp_s_ny = sir_pop_by_particle[:, 0, 0]
    tmp_i_ny = sir_pop_by_particle[:, 0, 1]
    tmp_r_ny = sir_pop_by_particle[:, 0, 2]
    tmp_n_ny = tmp_s_ny + tmp_i_ny + tmp_r_ny

    tmp_s_vt = sir_pop_by_particle[:, 0, 4]
    tmp_i_vt = sir_pop_by_particle[:, 0, 5]
    tmp_r_vt = sir_pop_by_particle[:, 0, 6]
    tmp_n_vt = tmp_s_vt + tmp_i_vt + tmp_r_vt

    # Compute seasonal transmission probability
    seasonal_multiplier = 1 + season * np.sin(2 * np.pi * (which_month + peak) / 12)
    infection_rate_ny = beta * seasonal_multiplier * tmp_i_ny / tmp_n_ny
    infection_prob_ny = 1 - np.exp(-infection_rate_ny)

    infection_rate_vt = beta * seasonal_multiplier * tmp_i_vt / tmp_n_vt
    infection_prob_vt = 1 - np.exp(-infection_rate_vt)

    # Draw infections and recoveries
    s_to_i_ny = rng.binomial(
        n=tmp_s_ny.astype(int),
        p=infection_prob_ny
    )
    s_to_i_vt = rng.binomial(
        n=tmp_s_vt.astype(int),
        p=infection_prob_vt
    )
    
    recovery_prob = 1 - np.exp(-gamma)
    i_to_r_ny = rng.binomial(
        n=tmp_i_ny.astype(int),
        p=recovery_prob
    )
    i_to_r_vt = rng.binomial(
        n=tmp_i_vt.astype(int),
        p=recovery_prob
    )

    # TODO: Impliment the births/deaths and other augmentations here...
    # Update states for next timestep
    sir_pop_by_particle[:, 1, 0] = tmp_s_ny - s_to_i_ny  # NY Susceptible
    sir_pop_by_particle[:, 1, 1] = tmp_i_ny + s_to_i_ny - i_to_r_ny  # infectious
    sir_pop_by_particle[:, 1, 2] = tmp_r_ny + i_to_r_ny  # Recovered
    sir_pop_by_particle[:, 1, 3] = s_to_i_ny  # new infections

    sir_pop_by_particle[:, 1, 4] = tmp_s_vt - s_to_i_vt  # Susceptible
    sir_pop_by_particle[:, 1, 5] = tmp_i_vt + s_to_i_vt - i_to_r_vt  # infectious
    sir_pop_by_particle[:, 1, 6] = tmp_r_vt + i_to_r_vt  # Recovered
    sir_pop_by_particle[:, 1, 7] = s_to_i_vt  # new infections

    # next timestep
    return sir_pop_by_particle[:, 1, :]


# TODO: determine what this function is doing exactly and adapt it to work with two populations and states
def run_smc(
    sir_out_last_month: np.ndarray,
    current_month: int,
    params: np.ndarray,
    observed_data: np.ndarray,
    rng: Generator,
) -> dict[str, np.ndarray]:
    """
    Run a Sequential Monte Carlo (SMC) step for the SIR model.

    :param sir_out_last_month: Latent states of the SIR model for each particle from the last month.
    :param current_month: The current month (0-11).
    :param params: Parameters for the SIR model in the following order:
        - beta: Transmission rate (log scale)
        - gamma: Recovery rate (log scale)
        - s_init_frac: Initial susceptible fraction (logit scale)
        - i_init_frac: Initial infected fraction (logit scale)
        - rho: Reporting rate (logit scale)
    :param observed_data: Observed cases for the current month.

    :return: A dictionary containing:
        - new_latent_states: Updated latent states of the SIR model for each particle.
        - new_log_likelihoods: Log likelihoods of the observed cases given predicted infections.
        - sampled_particles: Indices of the particles that were sampled.
    """
    rho = expit(params[4])
    
    # Run one timestep of the latent SIR model
    tmp_latent_states = run_one_timestep(
        current_latent_states=sir_out_last_month,
        which_month=current_month,
        params=params,
        rng=rng,
    )
    predicted_infections_ny = tmp_latent_states[:, 3].astype(int)  # new infections
    predicted_infections_vt = tmp_latent_states[:, 7].astype(int)  # new infections
    observed_cases_ny, observed_cases_vt = (observed_data[current_month, :]).astype(int)
    
    # TODO: Revisit this because it could be problematic to sum the tmp_likelihoods at this point...

    # Compute likelihood of observed cases given predicted infections
    tmp_likelihood_ny = np.array([
        binom.pmf(observed_cases_ny, pred, rho) if pred >= observed_cases_ny else 0.0
        for pred in predicted_infections_ny
    ])
    
    # If all likelihoods are zero, reset to a small constant
    if np.max(tmp_likelihood_ny) == 0:
        tmp_likelihood_ny = np.full(DEFAULT_NUM_PARTICLES, 1e-10)

    tmp_likelihood_vt = np.array([
        binom.pmf(observed_cases_vt, pred, rho) if pred >= observed_cases_vt else 0.0
        for pred in predicted_infections_vt
    ])
    
    # If all likelihoods are zero, reset to a small constant
    if np.max(tmp_likelihood_vt) == 0:
        tmp_likelihood_vt = np.full(DEFAULT_NUM_PARTICLES, 1e-10)

    tmp_likelihood = tmp_likelihood_ny + tmp_likelihood_vt

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


# TODO: This is similar to the function you wrote in the sir_aug.pu file called setup_sir_pop
def initialize_array(
    params: np.ndarray,
    population_ny: int,
    population_vt: int,
    num_time_steps: int,
    num_particles: int = DEFAULT_NUM_PARTICLES,
    num_us_states_in_model: int = DEFAULT_NUM_US_STATES_IN_MODEL,
) -> np.ndarray:
    """
    Initialize the SIR population array for the first month.

    :param params: Parameters for the SIR model in the following order:
        TODO: Update the order of params later
        - beta: Transmission rate (log scale)
        - gamma: Recovery rate (log scale)
        - season: Seasonality factor (logit scale)
        - peak: Peak month for seasonality (logit scale)
        - s_init_frac_ny: Initial susceptible fraction in New York (logit scale)
        - i_init_frac_ny: Initial infected fraction in New York (logit scale)
        - s_init_frac_vt: Initial susceptible fraction in Vermont (logit scale)
        - i_init_frac_vt: Initial infected fraction in Vermont (logit scale)
        - rho_ny: Reporting rate in New York (logit scale)
        - rho_vt: Reporting rate in Vermont (logit scale)
    :param population_ny: Population size for New York.
    :param population_vt: Population size for Vermont.
    :param num_time_steps: Number of time steps for the simulation.
    :param num_particles: Number of particles for the SMC algorithm.
    :param num_us_states_in_model: Number of US states included in the model (default is 2).
    
    :return: Initialized SIR population array with shape (num_particles, num_time_steps + 1, 4).
    """
    s_init_frac_ny = expit(params[4])
    i_init_frac_ny = expit(params[5]) * (1 - s_init_frac_ny)
    s_init_ny = round(s_init_frac_ny * population_ny)
    i_init_ny = round(i_init_frac_ny * population_ny)
    r_init_ny = population_ny - (s_init_ny + i_init_ny)

    s_init_frac_vt = expit(params[6])
    i_init_frac_vt = expit(params[7]) * (1 - s_init_frac_vt)
    s_init_vt = round(s_init_frac_vt * population_vt)
    i_init_vt = round(i_init_frac_vt * population_vt)
    r_init_vt = population_vt - (s_init_vt + i_init_vt)

    sir_out_all_months = np.full((num_particles, num_time_steps + 1, 4 * num_us_states_in_model), np.nan)

    # Set initial state
    sir_out_all_months[:, 0, 0] = s_init_ny  # Susceptible
    sir_out_all_months[:, 0, 1] = i_init_ny  # Infected
    sir_out_all_months[:, 0, 2] = r_init_ny  # Recovered
    sir_out_all_months[:, 0, 3] = 0    # New infections

    sir_out_all_months[:, 0, 4] = s_init_vt  # Susceptible
    sir_out_all_months[:, 0, 5] = i_init_vt  # Infected
    sir_out_all_months[:, 0, 6] = r_init_vt  # Recovered
    sir_out_all_months[:, 0, 7] = 0    # New infections

    return sir_out_all_months


# TODO: How to combine the log likelihood for different populations and states?
def calc_log_likelihood(
    params: np.ndarray,
    observed_data: np.ndarray,
    num_time_steps: int,
    rng: Generator,
) -> float:
    """
    Calculate the log likelihood of the SIR model given the parameters and observed data.

    :param params: Parameters for the SIR model in the following order:
        - beta: Transmission rate (log scale)
        - gamma: Recovery rate (log scale)
        - season: Seasonality factor (logit scale)
        - peak: Peak month for seasonality (logit scale)
        - s_init_frac: Initial susceptible fraction (logit scale)
        - i_init_frac: Initial infected fraction (logit scale)
        - rho: Reporting rate (logit scale)
    :param observed_data: Observed cases for each month.

    :return: Log likelihood of the observed data given the model parameters.
    """
    sir_out_all_months = initialize_array(
        params,
        population_ny=NEW_YORK_POP_1930,
        population_vt=VERMONT_POP_1930,
        num_time_steps=num_time_steps,
    )
    val = 0.0

    for month_step in range(1, num_time_steps + 1):
        tmp_out = run_smc(
            sir_out_last_month=sir_out_all_months[:, month_step - 1, :],
            current_month=month_step - 1,
            params=params,
            observed_data=observed_data,
            rng=rng,
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


def proposal_draw(proposal_standard_dev: np.ndarray, rng: Generator) -> np.ndarray:
    """
    Draw a proposal vector from a multivariate normal distribution.
     Using multivariate normal with mean 0 and diagonal covariance matrix.

    :param proposal_standard_dev: Standard deviations for each parameter in the proposal distribution.

    :return: A vector of proposed parameter changes.
    """
    num_params = len(proposal_standard_dev)
    vec = rng.multivariate_normal(mean=np.zeros(num_params), cov=np.diag(proposal_standard_dev**2))
    return vec


def propose_new_val(
    current_parameter_guess: np.ndarray,
    proposal_standard_dev: np.ndarray,
    observed_data: np.ndarray,
    num_time_steps: int,
    rng: Generator,
) -> dict[str, np.ndarray]:
    """
    Propose a new parameter guess based on the current guess and a proposal distribution.

    :param current_parameter_guess: Current guess for the parameters.
    :param proposal_standard_dev: Standard deviations for the proposal distribution.
    :param observed_data: Observed data to calculate the log likelihood.

    :return: A dictionary containing the new parameter guess and the log likelihood of the new guess.
    """
    new_parameter_guess = current_parameter_guess + proposal_draw(
        proposal_standard_dev,
        rng=rng
    )
    new_chain_info = calc_log_likelihood(
        params=new_parameter_guess,
        observed_data=observed_data,
        num_time_steps=num_time_steps,
        rng=rng
    )
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
    rng: Generator,
) -> dict[str, Any]:
    """
    Perform a Metropolis-Hastings step to decide whether to accept the new parameter guess.

    :param current_parameter_guess: Current guess for the parameters.
    :param new_parameter_guess: Proposed new guess for the parameters.
    :param current_log_likelihood: Log likelihood of the current parameter guess.
    :param new_log_likelihood: Log likelihood of the new parameter guess.
    :param current_latent: Current latent states of the SIR model.
    :param new_latent: Latent states of the SIR model for the new parameter guess.

    :return: A dictionary containing:
        - alpha: Acceptance ratio for the new parameter guess.
        - r_num: Random number drawn to decide acceptance.
        - accept_step: Boolean indicating whether the new guess was accepted.
        - next_parameter_guess: The parameter guess after the step.
        - next_latent: Latent states after the step.
        - next_log_likelihood: Log likelihood after the step.
        - new_parameter_guess: The proposed new parameter guess.
        - new_log_likelihood: Log likelihood of the proposed new guess.
    """
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


def run_one_mcmc_step(
    all_steps: dict,
    proposal_standard_dev: np.ndarray,
    observed_data: np.ndarray,
    num_time_steps: int,
    rng: Generator,
) -> dict:
    """
    Run one step of the MCMC algorithm using the Metropolis-Hastings method.

    :param all_steps: Dictionary containing the current state of the MCMC chain, latents, acceptance chain, and log likelihood.
        The keys should be:
        - "chain": Current parameter guesses (shape: (num_params, n_steps)).
        - "latents": Current latent states (list of arrays).
        - "accept_chain": Current acceptance chain (list of integers).
        - "log_likelihood": Current log likelihood (list of floats).
        - "current_parameter_guess": Current parameter guess (array).
        - "new_parameter_guess": Proposed new parameter guess (array).
        - "MH_info": Information from the last Metropolis-Hastings step (dict).
    :param proposal_standard_dev: Standard deviations for the proposal distribution.
    :param observed_data: Observed data to calculate the log likelihood.
    :param rng: Random number generator for reproducibility.

    :return: Updated dictionary with the new state of the MCMC chain, latents, acceptance chain, and log likelihood.
    """
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
        observed_data=observed_data,
        num_time_steps=num_time_steps,
        rng=rng,
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
        rng=rng,
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


def run_mcmc(
    n_steps: int,
    all_steps: dict,
    proposal_standard_dev: np.ndarray,
    observed_data: np.ndarray,
    num_time_steps: int,
    rng: Generator,
) -> dict:
    """
    Run the MCMC algorithm for a specified number of steps.

    :param n_steps: Number of MCMC steps to run.
    :param all_steps: Dictionary containing the initial state of the MCMC chain, latents, acceptance chain, and log likelihood.
        the keys should be:
        - "chain": Initial parameter guesses (shape: (num_params, 1)).
        - "latents": Initial latent states (list of arrays).
        - "accept_chain": Initial acceptance chain (list of integers).
        - "log_likelihood": Initial log likelihood (list of floats).
        - "current_parameter_guess": Current parameter guess (array).
        - "new_parameter_guess": Proposed new parameter guess (array).
        - "MH_info": Information from the last Metropolis-Hastings step (dict).
    :param proposal_standard_dev: Standard deviations for the proposal distribution.
    :param observed_data: Observed data to calculate the log likelihood.
    :param rng: Random number generator for reproducibility.

    :return: Updated dictionary with the final state of the MCMC chain, latents, acceptance chain, and log likelihood.
    """
    for _ in range(n_steps):
        all_steps = run_one_mcmc_step(
            all_steps,
            proposal_standard_dev,
            observed_data,
            num_time_steps=num_time_steps,
            rng=rng,
        )
    return all_steps
