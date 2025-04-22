from itertools import product
from typing import Iterable, Optional, Union
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit, logit

# np.set_printoptions(linewidth=1000, threshold=10000, precision=3)
np.random.seed(42)

POPULATION_PER_AGE = 1000
MAX_AGE = 50
NUM_YEARS = 50
GLOBAL_DEFAULT_A1 = 10
DEFAULT_Y_UNOBSERVED_YEARS = 10


class ForceOfInfectionMethod(Enum):
    """Enum for different methods of varying lambda."""
    YEAR = "year"
    AGE_PIECEWISE = "piecewise"
    AGE_LOG_LIN = "log_linear"

    def __str__(self):
        return self.value.replace("_", " ").title()


class InitialSusceptibleMethod(Enum):
    """Enum for different methods of varying initial susceptible."""
    BASIC = "basic"
    DIRECT = "direct"
    INDIRECT = "indirect"

    def __str__(self):
        return self.value.replace("_", " ").title()


class ParameterTransformMethod(Enum):
    """Enum for different methods of transforming parameters."""
    INT = "int"
    FLOAT = "float"

    def __str__(self):
        return self.value.replace("_", " ").title()


num_params_in_foi = {
    ForceOfInfectionMethod.YEAR: None,
    ForceOfInfectionMethod.AGE_PIECEWISE: 3,
    ForceOfInfectionMethod.AGE_LOG_LIN: 2,
}


def get_num_foi_params(foi_method: ForceOfInfectionMethod) -> Optional[int]:
    """Get the number of parameters in the force of infection function."""
    return num_params_in_foi.get(foi_method, None)


def get_infections_by_year(
    si_out: np.ndarray,
    num_years: int = NUM_YEARS,
) -> np.ndarray:
    """
    Get infections by year from the output of the catalytic model.

    :param si_out: Output of the catalytic model
    :param num_years: Number of years to consider

    :return: Infections by year
    """
    return si_out[2, :num_years, :MAX_AGE].sum(axis=1)


def get_subset_of_simulation_years(
    si_out: np.ndarray,
    unobserved_years: int = 0,
    observed_years: int = NUM_YEARS,
) -> np.ndarray:
    """
    Get a subset of the simulation years from the SI output.

    :param si_out: Output of the catalytic model
    :param unobserved_years: Index of the first unobserved year
    :param observed_years: Number of observed years

    :return: Subset of the simulation years
    """
    terminating_index = unobserved_years + observed_years
    return si_out[:, unobserved_years:terminating_index, :]


def sum_squared_error(
    ground_truth: np.ndarray,
    best_guess: np.ndarray,
) -> float:
    """
    Calculate the sum of squared errors between the ground truth and the best guess.

    :param ground_truth: The ground truth values
    :param best_guess: The guessed values

    :return: The sum of squared errors
    """
    return ((ground_truth - best_guess)**2).sum()


def transform_model_param(
    param: Union[float, int],
    method: ParameterTransformMethod,
    bounds: tuple[Union[int, float]] = None,
) -> Union[float, int, np.ndarray]:
    """
    Transform a model parameter based on the specified method.

    :param param: The parameter to transform.
    :param method: The transformation method to use.
    :param bounds: Optional bounds for the transformation.

    :return: The transformed parameter.
    """
    if method == ParameterTransformMethod.INT:
        if bounds:
            if len(bounds) == 2:
                max_int = bounds[1]
                param = (max_int * expit(param)).astype(int)
            elif len(bounds) == 1:
                param = np.exp(param).astype(int)
            else:
                raise ValueError("Invalid bounds provided. Must be a tuple of length 1 or 2.")
    elif method == ParameterTransformMethod.FLOAT:
        param = np.exp(param)
    else:
        raise NotImplementedError("No other transformation methods have been implemented yet.")
    
    return param


def evolve_disease_dynamics(
    si_out_current: np.ndarray,
    num_years_to_evolve: int,
    foi_by_year_and_age: np.ndarray,
    starting_year: int = 0,
    deterministic: bool = True,
    rho_observed_infections: float = 1.0,
) -> np.ndarray:
    """
    Evolve the disease dynamics for a given number of years.

    :param si_out_current: Current state of the disease dynamics
    :param num_years_to_evolve: Number of years to evolve the dynamics
    :param foi_by_year_and_age: Force of infection by year and age
    :param starting_year: Year to start evolving from
    :param deterministic: If True, use deterministic model
    :param rho_observed_infections: Ratio of observed infections to total infections

    :return: Evolved state of the disease dynamics
    """
    for y_num in range(starting_year, starting_year + num_years_to_evolve):
        for a_num in range(MAX_AGE):
            curr_susceptible = si_out_current[0, y_num, a_num]
            curr_infection = si_out_current[1, y_num, a_num]

            curr_lamb = foi_by_year_and_age[y_num, a_num]

            if deterministic:
                created_infections = rho_observed_infections * (curr_susceptible * (1 - np.exp(-curr_lamb)))
            else:
                # TODO: In the future make this a binomial random variable which uses 
                #   this as a probability for each susceptible person to be infected.
                raise NotImplementedError("Stochastic model not implemented yet.")

            new_susceptible = curr_susceptible - created_infections
            new_infection = curr_infection + created_infections

            si_out_current[0, y_num + 1, a_num + 1] = new_susceptible
            si_out_current[1, y_num + 1, a_num + 1] = new_infection
            si_out_current[2, y_num, a_num] = created_infections

    return si_out_current


def get_force_of_infection(
    force_of_infection_method: ForceOfInfectionMethod,
    model_params: np.ndarray,
    total_years: int,
) -> np.ndarray:
    """
    Get the force of infection by year and age based on the specified method.

    :param force_of_infection_method: Method of varying lambda
    :param model_params: Model parameters for the force of infection
    :param total_years: Total number of years to consider

    :return: Force of infection by year and age
    """
    if force_of_infection_method == ForceOfInfectionMethod.YEAR:
        model_params = transform_model_param(
            model_params,
            method=ParameterTransformMethod.FLOAT,
        )
        foi_by_year_and_age = np.repeat(model_params.reshape(-1, 1), MAX_AGE, axis=1)

    elif force_of_infection_method == ForceOfInfectionMethod.AGE_PIECEWISE:
        lambda_a = transform_model_param(model_params[0], method=ParameterTransformMethod.FLOAT)
        lambda_b = transform_model_param(model_params[1], method=ParameterTransformMethod.FLOAT)
        A_1 = transform_model_param(
            model_params[2],
            method=ParameterTransformMethod.INT,
            bounds=(0, MAX_AGE),
        )
        A_1 += 1

        lambda_by_age = np.zeros(MAX_AGE)
        lambda_by_age[:A_1] = lambda_a
        lambda_by_age[A_1:] = lambda_b

        foi_by_year_and_age = np.repeat(lambda_by_age.reshape(1, -1), total_years, axis=0)

    elif force_of_infection_method == ForceOfInfectionMethod.AGE_LOG_LIN:
        beta_0 = model_params[0]
        beta_1 = model_params[1]
        lambda_by_age = np.exp(beta_0 + beta_1 * np.arange(1, MAX_AGE + 1))
        foi_by_year_and_age = np.repeat(lambda_by_age.reshape(1, -1), total_years, axis=0)

    else:
        raise ValueError("Invalid force of infection method provided. No other variation methods have been implemented yet.")
    
    return foi_by_year_and_age


def initialize_susceptible(
    initial_susceptible_method: InitialSusceptibleMethod,
    y_unobserved_index: int,
    model_params: np.ndarray,
    si_out: np.ndarray,
    foi_by_year_and_age: np.ndarray,
    deterministic: bool,
    rho_observed_infections: float,    
) -> tuple[np.ndarray, int]:
    """
    Initialize the susceptible population based on the specified method.

    :param initial_susceptible_method: Method of initializing the susceptible population
    :param y_unobserved_index: Index of the unobserved years parameter
    :param model_params: Model parameters for the force of infection
    :param si_out: Current state of the disease dynamics
    :param foi_by_year_and_age: Force of infection by year and age
    :param deterministic: If True, use deterministic model
    :param rho_observed_infections: Ratio of observed infections to total infections

    :return: Updated state of the disease dynamics and the starting year for simulation
    """
    y_unobserved_years = transform_model_param(
        model_params[y_unobserved_index],
        method=ParameterTransformMethod.INT,
        bounds=(0,)
    )
    if initial_susceptible_method == InitialSusceptibleMethod.INDIRECT:
        si_out = evolve_disease_dynamics(
            si_out_current=si_out,
            starting_year=0,
            num_years_to_evolve=y_unobserved_years,
            foi_by_year_and_age=foi_by_year_and_age,
            deterministic=deterministic,
            rho_observed_infections=rho_observed_infections,
        )
        starting_year = y_unobserved_years

    elif initial_susceptible_method == InitialSusceptibleMethod.DIRECT:
        susc_by_age_start_index = y_unobserved_index + 1
        susceptible_by_age = transform_model_param(model_params[susc_by_age_start_index:], method=ParameterTransformMethod.FLOAT)
        si_out[0, y_unobserved_years, :-1] = susceptible_by_age
        starting_year = y_unobserved_years

    else:
        raise ValueError("Invalid initial susceptible method provided. No other init susceptible methods have been implemented yet.")

    return si_out, starting_year


def initialize_si_out(
    total_years: int,
    include_births: bool,
    population_per_age: int = POPULATION_PER_AGE,
    max_age: int = MAX_AGE,
) -> tuple[np.ndarray, int]:
    """
    Initialize the susceptible and infected populations tracking tensor

    :param total_years: Total number of years to consider
    :param include_births: Include births in the model
    :param population_per_age: Initial population per age group
    :param max_age: Maximum age to consider

    :return: Initialized susceptible and infected populations tensor
    """
    # susceptible and Infected by age group and year
    si_out = np.zeros((3, total_years+1, max_age+1))
    # Initial first year
    si_out[0, 0, :] = population_per_age
    if include_births:
        si_out[0, :, 0] = population_per_age # 1000 Births every year

    return si_out


def catalytic_model(
    model_params: np.ndarray,
    force_of_infection_method: ForceOfInfectionMethod,
    initial_susceptible_method: InitialSusceptibleMethod = InitialSusceptibleMethod.BASIC,
    num_years: int = NUM_YEARS,
    include_births: bool = True,
    deterministic: bool = True,
    rho_observed_infections: float = 1.0,
) -> np.ndarray:
    """
    Catalytic model for simulating disease dynamics.

    :param model_params: Parameters for the force of infection
    :param force_of_infection_method: Method of varying lambda
    :param initial_susceptible_method: Method of initializing the susceptible population
    :param num_years: Number of years to simulate
    :param include_births: Include births in the model
    :param deterministic: If True, use deterministic model
    :param rho_observed_infections: Ratio of observed infections to total infections

    :return: si_out: Susceptible and Infected by age group and year
    """
    if initial_susceptible_method == InitialSusceptibleMethod.BASIC:
        total_years = num_years
        foi_by_year_and_age = get_force_of_infection(
            force_of_infection_method=force_of_infection_method,
            model_params=model_params,
            total_years=total_years,
        )
        si_out = initialize_si_out(
            total_years=total_years,
            include_births=include_births
        )
        starting_year = 0

    elif initial_susceptible_method in [InitialSusceptibleMethod.DIRECT, InitialSusceptibleMethod.INDIRECT]:
        y_unobserved_index = get_num_foi_params(force_of_infection_method)
        y_unobserved_years = transform_model_param(
            model_params[y_unobserved_index],
            method=ParameterTransformMethod.INT,
            bounds=(0,)
        )
        total_years = num_years + y_unobserved_years
        foi_by_year_and_age = get_force_of_infection(
            force_of_infection_method=force_of_infection_method,
            model_params=model_params,
            total_years=total_years,
        )
        si_out = initialize_si_out(total_years=total_years, include_births=include_births)
        si_out, starting_year = initialize_susceptible(
            initial_susceptible_method=initial_susceptible_method,
            y_unobserved_index=y_unobserved_index,
            model_params=model_params,
            si_out=si_out,
            foi_by_year_and_age=foi_by_year_and_age,
            deterministic=deterministic,
            rho_observed_infections=rho_observed_infections,
        )

    else:
        raise ValueError("Invalid initial susceptible method provided. No other init susceptible methods have been implemented yet.")

    si_out = evolve_disease_dynamics(
        si_out_current=si_out,
        starting_year=starting_year,
        num_years_to_evolve=num_years,
        foi_by_year_and_age=foi_by_year_and_age,
        deterministic=deterministic,
        rho_observed_infections=rho_observed_infections,
    )
    si_out = get_subset_of_simulation_years(
        si_out=si_out,
        unobserved_years=starting_year,
        observed_years=num_years,
    )
    return si_out


def objective_function(
    model_params: np.ndarray,
    force_of_infection_method: ForceOfInfectionMethod,
    si_out_gt: np.ndarray,
    all_age_infection_error: bool = True,
    num_years: int = NUM_YEARS,
    initial_susceptible_method: InitialSusceptibleMethod = InitialSusceptibleMethod.BASIC,
    rho_observed_infections: float = 1.0,
    include_births: bool = True,
    deterministic: bool = True,
) -> float:
    """
    Objective function for optimization.

    :param model_params: Parameters to be optimized various forces of infection
    :param force_of_infection_method: Method of varying lambda
    :param si_out_gt: Ground truth output of the catalytic model
    :param all_age_infection_error: If True, calculate error for all ages
    :param num_years: Number of years to simulate
    :param initial_susceptible_method: Method of initializing the susceptible population
    :param rho_observed_infections: Ratio of observed infections to total infections
    :param include_births: Include births in the model
    :param deterministic: If True, use deterministic model

    :return: Sum of squared errors between ground truth and guessed infections
    """
    si_out = catalytic_model(
        model_params=model_params,
        force_of_infection_method=force_of_infection_method,
        initial_susceptible_method=initial_susceptible_method,
        num_years=num_years,
        include_births=include_births,
        deterministic=deterministic,
        rho_observed_infections=rho_observed_infections
    )

    if all_age_infection_error:
        infections_guess = get_infections_by_year(si_out, num_years=num_years)
        infections_gt = get_infections_by_year(si_out_gt, num_years=num_years)
    
    else:
        infections_guess = si_out[2, :num_years, :MAX_AGE]
        infections_gt = si_out_gt[2, :num_years, :MAX_AGE]
    
    # TODO: Fix how we are comparing the gt with the learned version... we need to only look at the final num_years years
    return sum_squared_error(infections_gt, infections_guess)


def get_params_initial_guess(
    foi_method: ForceOfInfectionMethod,
    init_susc_method: InitialSusceptibleMethod,
) -> np.ndarray:
    """
    Get the params for the optimization process
    """
    if foi_method == ForceOfInfectionMethod.AGE_PIECEWISE:
        # params = np.array([np.log(0.3), np.log(0.05), logit(GLOBAL_DEFAULT_A1/MAX_AGE), np.log(DEFAULT_Y_UNOBSERVED_YEARS)])
        params = np.array([np.log(0.1), np.log(0.1), logit(5/MAX_AGE), np.log(5)])
    
    elif foi_method == ForceOfInfectionMethod.AGE_LOG_LIN:
        # params = np.array([-1, -.1, np.log(DEFAULT_Y_UNOBSERVED_YEARS)])
        params = np.array([0, 0, np.log(5)])

    if init_susc_method == InitialSusceptibleMethod.DIRECT:
        direct_init_susc = np.insert(np.ones(MAX_AGE - 1)*np.log(900), 0, np.log(POPULATION_PER_AGE))
        params = np.concat([params, direct_init_susc])
    
    return params


def run_optimization(
    params: np.ndarray,
    foi_method: ForceOfInfectionMethod,
    si_out_gt: np.ndarray,
    all_age_infection_error: bool,
    num_years: int,
    init_susc_method: InitialSusceptibleMethod,
    rho_observed_infections: float = 1.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run the optimization process to fit the model parameters to the observed data.

    :param params: Initial guess for the model parameters.
    :param foi_method: The method used to calculate the force of infection.
    :param si_out_gt: The observed data (ground truth) for the model.
    :param all_age_infection_error: Flag to indicate if all age infection error should be considered.
    :param num_years: The number of years to simulate.
    :param init_susc_method: The method used to calculate the initial susceptible population.
    :param rho_observed_infections: The ratio of observed infections to total infections.

    :return: A tuple containing the predicted model output, optimized parameters, and the objective function value.
    """
    optim_results = minimize(
        objective_function,
        x0=params,
        args=(
            foi_method,
            si_out_gt,
            all_age_infection_error,
            num_years,
            init_susc_method,
            rho_observed_infections
        ),
        method="Nelder-Mead",
        # method="L-BFGS-B",
        options={"maxiter": 1000},
    )
    si_out_predicted = catalytic_model(
        model_params=optim_results.x,
        force_of_infection_method=foi_method,
        initial_susceptible_method=init_susc_method,
        num_years=num_years,
        include_births=True,
        deterministic=True,
    )
    return si_out_predicted, optim_results.x, optim_results.fun


def get_iteration_list() -> Iterable[tuple]: 
    configuration_iterator = product(
        [ForceOfInfectionMethod.AGE_PIECEWISE, ForceOfInfectionMethod.AGE_LOG_LIN],
        [15, 60],
        [InitialSusceptibleMethod.INDIRECT, InitialSusceptibleMethod.DIRECT],
    )
    return configuration_iterator
