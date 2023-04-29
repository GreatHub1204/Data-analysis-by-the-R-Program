function df = generate_data(df, V_CS, state_df, price_dist_steady, num_period, trans_mat_cum)    
  price_dist_steady_cumsum = cumsum(price_dist_steady); 
  price_id_consumer = 0;
  exceed_trans_prob_price = true;
  while exceed_trans_prob_price
      price_id_consumer = price_id_consumer + 1;
      if(df.eps_price_state_unif(1) > price_dist_steady_cumsum(price_id_consumer))
          exceed_trans_prob_price = true;
      elseif(df.eps_price_state_unif(1) < price_dist_steady_cumsum(price_id_consumer))
          exceed_trans_prob_price = false;
      end
  end
  df.state_id(1) = table2array(state_df(state_df.mileage_id == 1 & state_df.price_id == price_id_consumer, 'state_id')); 

  for t = 1:(num_period-1)
    
     % df.state_id(t) = table2array(state_df(state_df.mileage_id == 1 & state_df.price_id == price_id_consumer, 'state_id'));
    
    
    if df.state_id(t) ==0 || isnan(df.state_id(t))
        state_id_today = 1;
    else
        state_id_today = df.state_id(t);        
    end
    
    if table2array(V_CS(state_id_today, 'V_not_buy')) + df.eps_type1_not_buy(t) > table2array(V_CS(state_id_today, 'V_buy')) + df.eps_type1_buy(t)  
      
        df.action(t) = 0;            
        trans_mat_cum_today = trans_mat_cum.not_buy;
        % fprintf('trans_mat_cum_today = %f\n', trans_mat_cum_today);
    else     
        df.action(t) = 1;      
    
        trans_mat_cum_today = trans_mat_cum.buy;
    end      
    state_id_tomorrow = 0;
    exceed_trans_prob = true;
    break_flag = 0;
    while exceed_trans_prob
        state_id_tomorrow = state_id_tomorrow + 1;     
        
        trans_prob = trans_mat_cum_today(state_id_today, state_id_tomorrow);
        exceed_trans_prob = (df.eps_unif(t) > trans_prob);
        if state_id_tomorrow >125
            break_flag = 1;
            break;
        end
    end
    if break_flag ==1
        df.state_id(t+1) = 1;
        break_flag = 0;
    else
        df.state_id(t+1) = state_id_tomorrow;
    end
  end
end