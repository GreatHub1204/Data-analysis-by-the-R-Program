function EV = contraction(theta, beta, trans_mat, state_df, num_states, num_choice, Euler_const)
    U = flow_utility(theta, state_df);
    EV_old = zeros(num_states, num_choice);
    diff=1000;
    tol_level=1e-10;
    while(diff>tol_level)      
        EV_not_buy = Euler_const +trans_mat.not_buy * log(sum(exp(U + beta.*EV_old),2));
        EV_buy = Euler_const+trans_mat.buy * log(sum(exp(U + beta.*EV_old),2));
        EV_new = horzcat(EV_not_buy, EV_buy);       
        diff = sum(abs(EV_new - EV_old),"all");        
        EV_old = EV_new;
    end
    EV = EV_old;
    EV = array2table(EV,...
        'VariableNames',{'EV_not_buy','EV_buy'});
end