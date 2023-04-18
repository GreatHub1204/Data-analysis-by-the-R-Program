function df = generate_data(df, V_CS, state_df, price_dist_steady, eps_price_state_unif)
    
  price_dist_steady_cumsum = cumsum(price_dist_steady);
  
 
  price_id_consumer = 0;
  exceed_trans_prob_price = true;
  while exceed_trans_prob_price
      price_id_consumer = price_id_consumer + 1;
      exceed_trans_prob_price = ...
        (df.eps_price_state_unif(1) > price_dist_steady_cumsum(price_id_consumer));
  end
  

  df.state_id(1) = state_df(state_df.mileage_id == 1 & state_df.price_id == price_id_consumer, 'state_id');
  
 
  for t = 1:(num_period-1)

    state_id_today = df.state_id(t);
    
    
    if V_CS(state_id_today, 'V_not_buy') + df.eps_type1_not_buy(t) > ...
        V_CS(state_id_today, 'V_buy') + df.eps_type1_buy(t)
        
      
        df.action(t) = 0;
        
      
        trans_mat_cum_today = trans_mat_cum.not_buy;
    else
     
        df.action(t) = 1;
        
    
        trans_mat_cum_today = trans_mat_cum.buy;
    end
    
   
    state_id_tomorrow = 0;
    exceed_trans_prob = true;
    while exceed_trans_prob
        state_id_tomorrow = state_id_tomorrow + 1;
        trans_prob = trans_mat_cum_today(state_id_today, state_id_tomorrow);
        exceed_trans_prob = (df.eps_unif(t) > trans_prob);
    end
    df.state_id(t+1) = state_id_tomorrow;
  end
end