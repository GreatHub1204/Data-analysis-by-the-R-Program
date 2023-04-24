function loglh = logLH(theta, beta, trans_mat, state_df, df, num_states, num_choice, Euler_const)
    EV = contraction(theta, beta, trans_mat, state_df, num_states, num_choice, Euler_const);
    U = flow_utility(theta, state_df);
    V_CS = U + beta.*EV;
    prob_C = table2array(exp(V_CS))./table2array(sum(exp(V_CS),2));
    loglh = sum(log(mat_ij(df.state_id_state_df, df.action + 1, prob_C)));
end