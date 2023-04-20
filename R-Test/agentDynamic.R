rm(list = ls())

require(tidyverse)
require(skimr)

require(evd)

require(numDeriv)

library("plot3D")

library("magrittr")

library("dplyr")
library(purrr)

library("ggplot2")
library('base')
library(skimr)
library(tidyr)

theta_true <- c(theta_c = 0.004, theta_p = 0.003)   

beta <- 0.99

Euler_const <- - digamma(1)

Euler_const

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

mileage_trans_mat_true

mileage_trans_mat_true[1:4, 1:4, 1]

lambda_true <- c(0.1, 0.2, 0.2, 0.2, 0.2,
                 0.1, 0.2, 0.2, 0.2, 0.2,
                 0.1, 0.1, 0.2, 0.2, 0.1,
                 0.1, 0.1, 0.2, 0.2, 0.1,
                 0.05, 0.05, 0.1, 0.1, 0.2,
                 0.05, 0.05, 0.1, 0.1, 0.2)

price_trans_mat_true <- gen_price_trans(lambda_true)

price_trans_mat_true

trans_mat_true <- list()

mileage_trans_mat_true[,,1]
trans_mat_true$not_buy <- mileage_trans_mat_true[,,1] %x% price_trans_mat_true
trans_mat_true$buy <- mileage_trans_mat_true[,,2] %x% price_trans_mat_true


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

print(trans_mat_true$not_buy)










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
start_time

EV_true <- contraction(theta_true, beta, trans_mat_true, state_df)

end_time <- proc.time()
end_time

cat("Runtime:\n")

print((end_time - start_time)[[3]])


U_true <- flow_utility(theta_true, state_df)
U_true

V_CS_true <- U_true + beta*EV_true
colnames(V_CS_true) <- c("V_not_buy", "V_buy")

exp(V_CS_true[,"V_buy"])/rowSums(exp(V_CS_true))
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

print(data_gen)
data_gen


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
      if(state_id_tomorrow >125){
        break;
      }
    }
    df$state_id[t+1]<- state_id_tomorrow
  }
  return(df)
}


data_gen <- 
  data_gen %>%
  dplyr::group_split(consumer) %>%

  purrr::map_dfr(generate_data,
                 V_CS = V_CS_true,
                 state_df = state_df, 
                 price_dist_steady = price_dist_steady) %>% 

  dplyr::filter(period > (num_period - num_period_obs)) %>% 

  dplyr::left_join(state_df, by = 'state_id')

data_gen %>% tail(3)


rm(V_CS_true, trans_mat_cum)

data_gen %>% 
  dplyr::select(price, mileage, action) %>%
  skimr::skim() %>% 
  skimr::yank("numeric") %>% 
  dplyr::select(skim_variable, mean, sd, p0, p100) 
data_gen
  
data_gen %>%
  ggplot(aes(x = price)) + geom_histogram(binwidth = 100)


data_gen %>%
  ggplot(aes(x = mileage)) + geom_histogram(binwidth = 5)



data_gen %>% 
  dplyr::group_by(mileage) %>% 
  dplyr::summarize(num_state = n(),
                   sum_action = sum(action)) %>% 
  dplyr::mutate(prob_buy = sum_action / num_state) %>% 
  ggplot(aes(x = mileage, y = prob_buy)) + 
  geom_bar(stat = "identity")


data_gen %>% 
  dplyr::group_by(price) %>% 
  dplyr::summarize(num_state = n(),
                   sum_action = sum(action),
                   .groups = 'drop') %>% 
  dplyr::mutate(prob_buy = sum_action / num_state) %>% 
  ggplot(aes(x = price, y = prob_buy)) + 
  geom_bar(stat = "identity")

prob_buy_obs_mat <- 
  data_gen %>%
  dplyr::group_by(mileage,price) %>%
  dplyr::summarize(num_state = n(),
                   sum_action = sum(action),
                   .groups = 'drop') %>%
  dplyr::mutate(prob_buy = sum_action / num_state) %>% 
  dplyr::select(prob_buy) %>% 
  as.matrix() %>% 
  matrix(nrow = num_price_states, ncol = num_mileage_states)
prob_buy_obs_mat

hist3D(x = mileage_states, y = price_states, z = t(prob_buy_obs_mat), zlim=c(0,0.4),
       bty = "g", phi = 10,  theta = -60, axes=TRUE,label=TRUE,
       xlab = "Mileage", ylab = "Price", zlab = "Probability", main = "Conditional probability of buying",
       col = "#0080ff", border = "blue", shade = 0.4,
      ticktype = "detailed", space = 0.05, d = 2, cex.axis = 0.8)
data_gen

data_gen <- 
  data_gen %>% 
  dplyr::group_by(consumer) %>% 
  dplyr::mutate(lag_price_id = lag(price_id),
                lag_mileage_id = lag(mileage_id),
                lag_action = lag(action)) %>% 
    dplyr::ungroup() 


data_gen

num_cond_obs_mileage <- 
  data_gen %>% 
  # 1期目は推定に使えないため落とす
  dplyr::filter(period != (num_period - num_period_obs + 1)) %>% 
  # t期の走行距離、t+1期の走行距離、t期の購買ごとにグループ化して、観察数を数える
  dplyr::group_by(lag_mileage_id, mileage_id, lag_action) %>% 
  dplyr::summarise(num_cond_obs = n(),
                   .groups = 'drop') 
  

num_cond_obs_mileage 



kappa_est <- c()
kappa_est[1] <- 
  (num_cond_obs_mileage[2] * 
     (num_cond_obs_mileage[2] + num_cond_obs_mileage[3] + num_cond_obs_mileage[4])) /
  ((num_cond_obs_mileage[2] + num_cond_obs_mileage[3]) * 
     (num_cond_obs_mileage[1] + num_cond_obs_mileage[2] + 
        num_cond_obs_mileage[3] + num_cond_obs_mileage[4]))
kappa_est[2] <- 
  (num_cond_obs_mileage[3] * 
     (num_cond_obs_mileage[2] + num_cond_obs_mileage[3] + num_cond_obs_mileage[4])) /
  ((num_cond_obs_mileage[2] + num_cond_obs_mileage[3]) * 
     (num_cond_obs_mileage[1] + num_cond_obs_mileage[2] + 
        num_cond_obs_mileage[3] + num_cond_obs_mileage[4]))

Infomat_mileage_est <- matrix(0, nrow = 2, ncol = 2)

# 最尤法のフィッシャー情報量を求める
Infomat_mileage_est[1,1] <- 
  (num_cond_obs_mileage[1] / (1 - kappa_est[1] - kappa_est[2])^2) +
  (num_cond_obs_mileage[2] / kappa_est[1]^2) +
  (num_cond_obs_mileage[4] / (kappa_est[1]+kappa_est[2])^2)
Infomat_mileage_est[1,2] <- 
  (num_cond_obs_mileage[1] / (1 - kappa_est[1] - kappa_est[2])^2) +
  (num_cond_obs_mileage[4] / (kappa_est[1]+kappa_est[2])^2)
Infomat_mileage_est[2,1] <- Infomat_mileage_est[1,2]
Infomat_mileage_est[2,2] <- 
  (num_cond_obs_mileage[1] / (1 - kappa_est[1] - kappa_est[2])^2) +
  (num_cond_obs_mileage[3] / kappa_est[2]^2) +
  (num_cond_obs_mileage[4] / (kappa_est[1]+kappa_est[2])^2)

# 逆行列の対角要素の平方根が標準誤差になる
kappa_se <- sqrt(diag(solve(Infomat_mileage_est)))

dplyr::tibble(kappa_est, kappa_se)

num_cond_obs_price <- 
  data_gen %>% 
  # 1期目は推定に使えないため落とす
  dplyr::filter(period != (num_period - num_period_obs + 1)) %>% 
  # t期の価格、t+1期の価格ごとにグループ化して、観察数を数える
  dplyr::group_by(lag_price_id, price_id) %>% 
  dplyr::summarise(num_cond_obs = n(),
                   .groups = 'drop') %>% 
  # 観察数を行列（num_price_states行の正方行列）に変換
  # price_id (t+1期の価格) を横に広げる
  tidyr::pivot_wider(names_from = "price_id",
                     values_from = "num_cond_obs") %>%
  dplyr::select(!lag_price_id) %>% 
  as.matrix()

lambda_est_mat <- 
  num_cond_obs_price / rowSums(num_cond_obs_price)
lambda_est_mat


lambda_se <- c()
for (i in 1:num_price_states) {
  # 最尤法のフィッシャー情報量を求める
  Infomat_price_est <- 
    diag(num_cond_obs_price[i,],
         num_price_states)[-i,-i] / 
    (lambda_est_mat[-i,-i] ^ 2) + 
    (num_cond_obs_price[i,i] / 
       lambda_est_mat[i,i] ^ 2) *
    matrix(1, num_price_states, num_price_states)[-i,-i]
  lambda_se <- c(
    lambda_se,
    # 逆行列の対角要素の平方根が標準誤差になる
    sqrt(diag(solve(Infomat_price_est)))
  )
}

lambda_se_mat <- 
  c(0, lambda_se[1], lambda_se[2], lambda_se[3], lambda_se[4], lambda_se[5],
    lambda_se[6], 0, lambda_se[7], lambda_se[8], lambda_se[9], lambda_se[10],
    lambda_se[11], lambda_se[12], 0, lambda_se[13], lambda_se[14], lambda_se[15],
    lambda_se[16], lambda_se[17], lambda_se[18], 0, lambda_se[19], lambda_se[20],
    lambda_se[21], lambda_se[22], lambda_se[23], lambda_se[24], 0, lambda_se[25],
    lambda_se[26], lambda_se[27], lambda_se[28], lambda_se[29], lambda_se[30], 0) %>% 
  matrix(ncol = num_price_states, nrow = num_price_states, byrow=T)
lambda_se_mat


lambda_est <- as.vector(t(lambda_est_mat))[c(-1,-8,-15,-22,-29,-36)]
dplyr::tibble(lambda_est, lambda_se)

mat_ij <- Vectorize(
  function(i,j,mat) {mat[i,j]},
  vectorize.args = c("i", "j"))


logLH_stat <- function(theta, state_df, df){
  
  
  # 選択毎の効用関数を求める
  U <- flow_utility(theta, state_df)
  # 選択確率を計算
  prob_C_stat <- exp(U) / rowSums(exp(U))
  # 対数尤度を計算
  sum(log(mat_ij(df$state_id, df$action + 1, prob_C_stat)))
}


start_time <- proc.time()

# 最適化
logit_stat_opt <- optim(theta_true, logLH_stat,
                        state_df = state_df, df = data_gen, 
                        control = list(fnscale = -1), 
                        method = "Nelder-Mead")

end_time <- proc.time()
cat("Runtime:\n")

print((end_time - start_time)[[3]])


theta_est_stat <- logit_stat_opt$par
theta_est_stat


hessian_stat <- numDeriv::hessian(func = logLH_stat, x = theta_est_stat, 
                                  state_df = state_df, df = data_gen)
theta_se_stat <- sqrt(diag(solve(-hessian_stat)))
dplyr::tibble(theta_est_stat, theta_se_stat)


trans_mat_hat <- list()
trans_mat_hat$not_buy <- 
  gen_mileage_trans(kappa_est)[,,1] %x% gen_price_trans(lambda_est)
trans_mat_hat$buy <- 
  gen_mileage_trans(kappa_est)[,,2] %x% gen_price_trans(lambda_est)


logLH <- function(theta, beta, trans_mat, state_df, df){
  
  # 選択ごとの期待価値関数を計算
  EV <- contraction(theta, beta, trans_mat, state_df)
  
  # 選択毎の価値関数を定義する
  U <- flow_utility(theta, state_df)
  V_CS <- U + beta*EV
  # 選択確率を計算
  prob_C <- exp(V_CS) / rowSums(exp(V_CS))
  # 対数尤度を計算
  sum(log(mat_ij(df$state_id, df$action + 1, prob_C)))
}


start_time <- proc.time()

# 最適化
NFXP_opt <- optim(theta_true, logLH,
                  beta = beta, trans_mat = trans_mat_hat, state_df = state_df, df = data_gen, 
                  control = list(fnscale = -1), 
                  method = "Nelder-Mead")

end_time <- proc.time()
cat("Runtime:\n")

print((end_time - start_time)[[3]])

theta_est <- NFXP_opt$par
theta_est


hessian <- numDeriv::hessian(func = logLH, x = theta_est, 
                             beta = beta,　trans_mat = trans_mat_hat, state_df = state_df, df = data_gen)
theta_se <- sqrt(diag(solve(-hessian)))
dplyr::tibble(theta_est, theta_se)
