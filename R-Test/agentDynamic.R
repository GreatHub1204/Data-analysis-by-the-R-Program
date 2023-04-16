rm(list = ls())

require(tidyverse)
require(skimr)

require(evd)

require(numDeriv)

library("plot3D")

library("magrittr")

library("dplyr")

library("ggplot2")

theta_true <- c(theta_c = 0.004, theta_p = 0.003)   

beta <- 0.99

Euler_const <- - digamma(1)


num_choice <- 2

price_states <- seq(2000, 2500, by = 100)

mileage_states <- seq(0, 100, by = 5)

num_price_states <- length(price_states)

num_mileage_states <- length(mileage_states)

num_states <- num_price_states * num_mileage_states

state_df <- dplyr::tibble(
  state_id = 1:num_states,
  price_id = rep(1:num_price_states, num_mileage_states),
  mileage_id = rep(1:num_mileage_states, each = num_price_states),
  price = rep(price_states, times = num_mileage_states),
  mileage = rep(mileage_states, each = num_price_states)
)




state_df %>% tail(3)

gen_mileage_trans <- function(kappa){
  kappa_1 <- kappa[1]
  kappa_2 <- kappa[2]
  
  mileage_trans_mat_hat_not_buy <- matrix(0, ncol = num_mileage_states, nrow = num_mileage_states)
  for (i in 1:num_mileage_states) {
    for(j in 1:num_mileage_states){
      if(i == j){
        mileage_trans_mat_hat_not_buy[i,j] <- 1-kappa_1 - kappa_2
      }else if(i ==j-1){
        mileage_trans_mat_hat_not_buy[i, j] <- kappa_1
      }else if(i == j -1){
        mileage_trans_mat_hat_not_buy[i, j] <- kappa_2
      }
    }
  }
  
  mileage_trans_mat_hat_not_buy[num_mileage_states - 1, num_mileage_states] <- kappa_1 + kappa_2
  mileage_trans_mat_hat_not_buy[num_mileage_states, num_mileage_states] <- 1
  
  
  mileage_trans_mat_hat_buy <- matrix(1, nrow = num_mileage_states, ncol = 1) %*% mileage_trans_mat_hat_not_buy[1,]
  
  return(array(c(mileage_trans_mat_hat_not_buy,
                 mileage_trans_mat_hat_buy),
               dim = c(num_mileage_states, num_mileage_states, num_choice)))
}

gen_price_trans <- function(lambda){
  lambda_11 <- 1 - lambda[1] - lambda[2] - lambda[3] - lambda[4] - lambda[5]
  lambda_22 <- 1 - lambda[6] - lambda[7] - lambda[8] - lambda[9] - lambda[10]
  lambda_33 <- 1 - lambda[11] - lambda[12] - lambda[13] - lambda[14] - lambda[15]
  lambda_44 <- 1 - lambda[16] - lambda[17] - lambda[18] - lambda[19] - lambda[20]
  lambda_55 <- 1 - lambda[21] - lambda[22] - lambda[23] - lambda[24] - lambda[25]
  lambda_66 <- 1 - lambda[26] - lambda[27] - lambda[28] - lambda[29] - lambda[30]
  price_trans_mat_hat <- 
    c(lambda_11, lambda[1], lambda[2], lambda[3], lambda[4], lambda[5],
      lambda[6], lambda_22, lambda[7], lambda[8], lambda[9], lambda[10],
      lambda[11], lambda[12], lambda_33, lambda[13], lambda[14], lambda[15],
      lambda[16], lambda[17], lambda[18], lambda_44, lambda[19], lambda[20],
      lambda[21], lambda[22], lambda[23], lambda[24], lambda_55, lambda[25],
      lambda[26], lambda[27], lambda[28], lambda[29], lambda[30], lambda_66) %>% 
    matrix(ncol = num_price_states, nrow = num_price_states, byrow=T)
  return(price_trans_mat_hat)
}

kappa_true <- c(0.25, 0.05)

mileage_trans_mat_true <- gen_mileage_trans(kappa_true)



mileage_trans_mat_true[1:4, 1:4, 1]

lambda_true <- c(0.1, 0.2, 0.2, 0.2, 0.2,
                 0.1, 0.2, 0.2, 0.2, 0.2,
                 0.1, 0.1, 0.2, 0.2, 0.1,
                 0.1, 0.1, 0.2, 0.2, 0.1,
                 0.05, 0.05, 0.1, 0.1, 0.2,
                 0.05, 0.05, 0.1, 0.1, 0.2)

price_trans_mat_true <- gen_price_trans(lambda_true)


trans_mat_true <- list()

mileage_trans_mat_true[,,2]

trans_mat_true$not_buy <- mileage_trans_mat_true[,,1] %x% price_trans_mat_true
trans_mat_true$buy <- mileage_trans_mat_true[,,2] %x% price_trans_mat_true
mileage_trans_mat_true[,,2]

price_trans_eigen <- eigen(t(price_trans_mat_true))
price_dist_steady <- price_trans_eigen$vectors[,1]/sum(price_trans_eigen$vectors[,1])


flow_utility <- function(theta, state_df){
  theta_c <- theta[1]
  theta_p <- theta[2]
  U <- 
    cbind(
      
      U_not_buy = - theta_c * state_df$mileage, 
      
      
      U_buy = - theta_p * state_df$price
    ) 
  return(U)
}

contraction <- 
  function(theta, beta, trans_mat, state_df) {

    U <- flow_utility(theta, state_df)

    EV_old <- matrix(0, nrow = num_states, ncol = num_choice)

    diff <- 1000
 
    tol_level <- 1.0e-10
    
    while (diff > tol_level) {

      EV_new <- cbind(
        EV_not_buy <- 
          Euler_const + trans_mat$not_buy %*% log(rowSums(exp(U + beta*EV_old))),
        EV_buy <-
          Euler_const + trans_mat$buy %*% log(rowSums(exp(U + beta*EV_old)))
      )

      diff <- sum(abs(EV_new-EV_old))
      

      EV_old <- EV_new
    }
    EV <- EV_old
    colnames(EV) <- c("EV_not_buy", "EV_buy")
    return(EV)
  }

start_time <- proc.time()

EV_true <- contraction(theta_true, beta, trans_mat_true, state_df)

end_time <- proc.time()

cat("Runtime:\n")

print((end_time - start_time)[[3]])


U_true <- flow_utility(theta_true, state_df)
V_CS_true <- U_true + beta*EV_true
colnames(V_CS_true) <- c("V_not_buy", "V_buy")

prob_buy_true_mat <- matrix(exp(V_CS_true[,"V_buy"])/rowSums(exp(V_CS_true)), 
                            nrow = num_price_states, ncol = num_mileage_states)
prob_buy_true_mat

num_consumer <- 1000

num_period <- 12 * 50
num_period_obs <- 12 * 10

num_obs <- num_consumer * num_period

trans_mat_cum <- list()
trans_mat_cum$not_buy <- t(apply(trans_mat_true$not_buy, 1, cumsum))
trans_mat_cum$buy <- t(apply(trans_mat_true$buy, 1, cumsum))


set.seed(1)

data_gen <- 
  dplyr::tibble(
    consumer = rep(1:num_consumer, each = num_period),
    period = rep(1:num_period, times = num_consumer),
    eps_type1_not_buy = evd::rgev(num_obs),
    eps_type1_buy = evd::rgev(num_obs),
    eps_unif = runif(num_obs),
    eps_price_state_unif = runif(num_obs),
    state_id = 0,
    action = 0
  )

generate_data <- function(df, V_CS, state_df, price_dist_steady) {
  

  price_dist_steady_cumsum <- cumsum(price_dist_steady)

  price_id_consumer <- 0
  exceed_trans_prob_price <- TRUE
  while(exceed_trans_prob_price) {
    price_id_consumer <- price_id_consumer + 1
    exceed_trans_prob_price <- 
      (df$eps_price_state_unif[1] >
         price_dist_steady_cumsum[price_id_consumer])
  }
  

  df$state_id[1] <-  state_df %>% 
    dplyr::filter(mileage_id == 1) %>% 
    dplyr::filter(price_id == price_id_consumer) %>% 
    dplyr::select(state_id) %>% 
    as.numeric()
  

  for (t in 1:(num_period-1)) {

    state_id_today <- df$state_id[t]
    

    if (V_CS[,'V_not_buy'][state_id_today] + df$eps_type1_not_buy[t] > 
        V_CS[,'V_buy'][state_id_today] + df$eps_type1_buy[t]){
      

      df$action[t] <- 0
      

      trans_mat_cum_today <- trans_mat_cum$not_buy
      
    }else{

      df$action[t] <- 1

      trans_mat_cum_today <- trans_mat_cum$buy
      
    }
    

    state_id_tomorrow <- 0
    exceed_trans_prob <- TRUE

    while (exceed_trans_prob) {
      state_id_tomorrow <- state_id_tomorrow + 1
      trans_prob <- trans_mat_cum_today[state_id_today, state_id_tomorrow]
      exceed_trans_prob <- (df$eps_unif[t] > trans_prob)
    }
    df$state_id[t+1]<- state_id_tomorrow
  }
  return(df)
}
data_gen%>%select(2)

data_gen <- 
  data_gen %>%
  group_split(consumer) %>%

  purrr::map_dfr(generate_data,
                 V_CS = V_CS_true,
                 state_df = state_df, 
                 price_dist_steady = price_dist_steady) %>% 

  filter(period > (num_period - num_period_obs)) %>% 

  left_join(state_df, by = 'state_id')

data_gen %>% tail(3)

##rm(V_CS_true, trans_mat_cum)

data_gen %>% 
  dplyr::select(price, mileage, action) %>%
  skimr::skim() %>% 
  skimr::yank("numeric") %>% 
  dplyr::select(skim_variable, mean, sd, p0, p100) 
data_gen %>%
  ggplot(aes(x = price)) + geom_histogram(binwidth = 100)



