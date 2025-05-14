rm(list = ls())
set.seed(1)
require(glue)
require(RColorBrewer)

# Constants
Population <- 1000
NumYears <- 1
Year_start <- 2000
Year_end <- Year_start + NumYears - 1
Date_start <- as.Date(glue("{Year_start}-01-01"))
Date_end <- as.Date(glue("{Year_end}-12-31"))
time_step = "1 week"
Dates <- seq(Date_start, Date_end, by = time_step)
dt <- ifelse(time_step == "1 year", 1, ifelse(time_step == "1 month", 1/12, ifelse(time_step == "1 week", 1/52, 1/365)))

beta <- function(date, par){
  return(par)
}

run_SIR <- function(pars){
  SIROut <- array(0, dim = c(length(Dates) + 1, 6))
  # [,1]: Susceptible
  # [,2]: Infectious
  # [,3]: Recovered
  # [,4]: new_infections
  # [,5]: new_recovered
  # [,6]: new_cases
  
  # SIROut[t,1] + SIROut[t,2] + SIROut[t,3] = Population
  SIROut[1,1] <- pars[1]
  SIROut[1,2] <- pars[5]
  SIROut[1,3] <- Population - SIROut[1,1] - SIROut[1,2]
  beta <- pars[2]
  gamma <- pars[3]
  rho <- pars[4]
  
  for (d_num in seq_along(Dates)){
    date <- Dates[d_num]
    # Get our current compartments
    current_S <- SIROut[d_num, 1]
    current_I <- SIROut[d_num, 2]
    current_R <- SIROut[d_num, 3]
    current_N <- current_S + current_I + current_R
    # Calculate transition probabilities
    p_inf <- 1 - exp(-beta * (current_I / current_N) * dt)
    p_rec <- 1 - exp(-gamma * dt)
    # Calculate number of transitions
    new_infections <- rbinom(1, current_S, p_inf)
    new_recovered <- rbinom(1, current_I, p_rec)
    new_cases <- rbinom(1, new_infections, rho)
    # Update SIROut
    SIROut[d_num + 1, 1] <- current_S - new_infections
    SIROut[d_num + 1, 2] <- current_I + new_infections - new_recovered
    SIROut[d_num + 1, 3] <- current_R + new_recovered
    SIROut[d_num, 4] <- new_infections
    SIROut[d_num, 5] <- new_recovered
    SIROut[d_num, 6] <- new_cases
    
  }
  toss <- length(Dates) + 1
  SIROut <- SIROut[-toss,]
  return(SIROut)
}

#
true_pars <- c(?, ?, ?, ?)
truth_out <- run_SIR(pars = true_pars)
obs_data <- truth_out[,6]

error <- function(pars, max_attempts = 5){
  attempt <- 0
  valid_ll <- FALSE
  
  while (attempt < max_attempts & !valid_ll){
    attempt <- attempt + 1
    guess_out <- run_SIR(pars)
    p_inf <- 1 - exp(-pars[2] * guess_out[,2] / rowSums(guess_out[,1:3]) * dt)
    p_case <- p_inf * pars[4]
    # if (sum(guess_out[,1] >= obs_data) == length(obs_data)){
    if (length(which(guess_out[,1] < obs_data)) == 0){
      log_lik <- sum(dbinom(x = obs_data, size = guess_out[,1], prob = p_case, log = TRUE))
      valid_ll <- TRUE
    }
  }
  if (!valid_ll) log_lik <- -1e20
  
  return(log_lik)
}







#### Example Plot #1
COLS <- brewer.pal(4, "Set1")
COLS <- cbind(COLS,adjustcolor(COLS, alpha = 0.1))

n_draws <- 25
best_guess_pars <- c(990, 365/10, 365/70, 1, 10)
draw_results <- lapply(seq_len(n_draws), function(x) run_SIR(pars = best_guess_pars))

obs_data <- draw_results[[1]][,6] # You need to replace this with the provided case data for each part

best_ll <- error(best_guess_pars)

par(mfrow=c(2,1), mar = c(5.1,5.1,1.1,0.6), oma = c(0,0,2.1,0))
plot(Dates, rep(1,length(Dates)), type = 'n', 
     xlab = "Date", ylab = "Individuals", ylim = c(0,Population))
for (n_num in seq_len(n_draws)){
  tmp_draw_results <- draw_results[[n_num]]
  lines(Dates, tmp_draw_results[,1], col = COLS[2,2], lwd = 2)
  lines(Dates, tmp_draw_results[,2], col = COLS[1,2], lwd = 2)
  lines(Dates, tmp_draw_results[,3], col = COLS[3,2], lwd = 2)
}
legend("right", legend = c("Susceptible", "Infectious", "Recovered"), lwd = 2, col = COLS[1:3,1], bty = "n")

XMAX <- max(which(obs_data > 0))

plot(Dates[seq_len(XMAX)], obs_data[seq_len(XMAX)], type = 'n', 
     xlab = "Date", ylab = "Cases", ylim = c(0,1.25*max(Fake_Cases)))
for (n_num in seq_len(n_draws)){
  tmp_draw_results <- draw_results[[n_num]]
  lines(Dates[seq_len(XMAX)], tmp_draw_results[seq_len(XMAX),6], col = COLS[4,2], lwd = 2)
}
points(Dates[seq_len(XMAX)], obs_data[seq_len(XMAX)], pch = 21, cex = 2, bg = COLS[4,1])
legend("topright", legend = c("Observed", "Predicted"), lwd = c(NA,2), pch = c(21, NA), 
       col = c(1,COLS[4,1]), pt.bg = c(COLS[4,1],NA), pt.cex = c(2, NA), bty = "n")

mtext(glue("Best log-likelihood for Dataset 1: {best_ll}"), 3, outer = TRUE)
