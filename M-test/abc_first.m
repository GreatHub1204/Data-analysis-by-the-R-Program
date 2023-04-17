
theta_c = 0.004;
theta_p = 0.003;
theta_true = [theta_c, theta_p];
beta = 0.99;
Euler_const = -psi(1);


num_choice = 2;
price_states = 2000:100:2500;
mileage_states = 0:5:100;

num_price_states = length(price_states);

num_mileage_states = length(mileage_states);


num_states = num_price_states * num_mileage_states;


state_id = (1:num_states)';
price_id = repmat(1:num_price_states, [1, num_mileage_states]);
mileage_id = repmat(1:num_mileage_states, [num_price_states, 1]);
price = repmat(price_states, [1, num_mileage_states])';
mileage = repmat(mileage_states, [num_price_states, 1]);

state_df = table(state_id, price_id(:), mileage_id(:), price(:), mileage(:));
state_df.Properties.VariableNames = {'state_id', 'price_id', 'mileage_id', 'price', 'mileage'};

state_df(end-2:end,:)
format short
kappa_true = [0.25, 0.05];
kappa_true

mileage_trans_mat_true = gen_mileage_trans(kappa_true, num_mileage_states, num_choice);

mileage_trans_mat_true(1:4,1:4,1);

lambda_true = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.05, 0.05, 0.1, 0.1, 0.2, 0.05, 0.05, 0.1, 0.1, 0.2];
price_trans_mat_true = gen_price_trans(lambda_true);

trans_mat_true = [];
trans_mat_true.not_buy = kron(mileage_trans_mat_true(:,:,1), price_trans_mat_true);
trans_mat_true.buy = kron(mileage_trans_mat_true(:,:,2), price_trans_mat_true);


price_trans_eigen = eig(price_trans_mat_true.');

[V,D] = eig(price_trans_mat_true.');

price_dist_steady = V(:,1)/sum(V(:,1));

start_time = toc;



% EV_true = contraction(theta_true, beta, trans_mat_true, state_df, num_states, num_choice, Euler_const);


% end_time = toc
% disp('Runtime:')
% disp(end_time - start_time);
% U_true = flow_utility(theta_true, state_df);




