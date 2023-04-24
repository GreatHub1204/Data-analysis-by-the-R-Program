function U = flow_utility(theta, state_df)
    theta_c = theta(1);
    
    theta_p = theta(2);
    U_not_buy = - theta_c.*state_df.mileage;
    U_boy = - theta_p.*state_df.price;
    U = [U_not_buy U_boy];
end
