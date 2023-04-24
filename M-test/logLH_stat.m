function lh_value = logLH_stat(theta, state_df, df)

    U = flow_utility(theta, state_df);

    prob_C_stat = exp(U) ./ sum(exp(U),2);
    lh_value =sum(log(mat_ij(df.state_id_state_df, df.action+1, prob_C_stat)));
end

