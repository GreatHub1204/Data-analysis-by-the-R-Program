function scalar = logLH_stat_test(theta, state_df, df)
    scalar = 0;
    U = flow_utility(theta, state_df);

    prob_C_stat = exp(U)./sum(exp(U),2);
     
    % fprintf('prob_C_stat = %f\n', prob_C_stat);
    scalar = sum(log(mat_ij_test(df.state_id_state_df, df.action+1, prob_C_stat)));

    
    % fprintf('gradient = %f\n', gradient);
    if ~isfinite(gradient)
        error('Function value is not finite.');
    end
    
end