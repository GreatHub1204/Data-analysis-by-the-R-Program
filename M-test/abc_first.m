


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

EV_true = contraction(theta_true, beta, trans_mat_true, state_df, num_states, num_choice, Euler_const);

end_time = toc;

disp('Runtime:')
disp(end_time - start_time)

U_true = flow_utility(theta_true, state_df);

V_CS_true = U_true + beta.*EV_true;
V_CS_true = renamevars(V_CS_true,["EV_not_buy","EV_buy"],["V_not_buy","V_buy"]);


exp_V_buy = exp(V_CS_true(:, "V_buy"));
exp_V_buy = renamevars(exp_V_buy,["V_buy"],["V1"]);
exp_V_sum = sum(exp(V_CS_true),2);
exp_V_sum = renamevars(exp_V_sum,["sum"],["V1"]);
format long
prob_buy = exp_V_buy ./ exp_V_sum;
prob_buy;
prob_buy1 = table2array(prob_buy)

trans_mat_true_not_buy = reshape(prob_buy1, [num_price_states, num_mileage_states]);

trans_mat_true_not_buy;

num_consumer = 1000;

num_period = 12 * 50;

num_period_obs = 12*10;

num_obs = num_consumer * num_period;

trans_mat_cum = [];
format short
trans_mat_cum.not_buy = cumsum(trans_mat_true.not_buy, 2);

trans_mat_cum.buy = cumsum(trans_mat_true.buy, 2);


rng(1)


consumer = repmat(1:num_consumer, [num_period, 1]);
period = repmat(1:num_period,[1, num_consumer]);
eps_type1_not_buy = gevrnd(0, 1, 0, num_obs, 1);
eps_type1_buy = gevrnd(0, 1, 0, num_obs, 1);
eps_unif = rand(num_obs, 1);
eps_price_state_unif = rand(num_obs,1);
state_id = zeros(num_obs, 1);
action = zeros(num_obs, 1);
 
% data_gen = table(consumer(:), period(:),eps_type1_not_buy(:), eps_type1_buy(:),  eps_unif(:), eps_price_state_unif(:), state_id, action);
% data_gen.Properties.VariableNames = {'consumer', 'period','eps_type1_not_buy', 'eps_type1_buy', 'eps_unif', 'eps_price_state_unif', 'state_id', 'action'};
% 
% data_gen_groups = splitapply(@ (consumer, period, eps_type1_not_buy, eps_type1_buy, eps_unif, eps_price_state_unif, state_id, action) {table(consumer, period, eps_type1_not_buy, eps_type1_buy, eps_unif, eps_price_state_unif, state_id, action)}, data_gen, findgroups(data_gen.consumer));
% 
% results = cell(1, numel(data_gen_groups));
% 
% for i = 1:numel(data_gen_groups)
%     group = data_gen_groups{i};
% 
%     results{i} = generate_data(group, V_CS_true, state_df, price_dist_steady, num_period, trans_mat_cum);
%      data_gen_result = vertcat(results{:});
% end
% 
% data_gen_result = data_gen_result(data_gen_result.period > (num_period - num_period_obs), :);
% data_gen_result = outerjoin(data_gen_result, state_df, 'Keys', 'state_id');
% tail(data_gen_result, 3)

clear V_CS_true trans_mat_cum;

data_gen_selected = data_gen_result(:, {'price', 'mileage', 'action'});
mean_val = mean(data_gen_selected);
mean_val = reshape(table2array(mean_val), [3,1])
sd_val = std(data_gen_selected);
sd_val = reshape(table2array(sd_val), [3,1])
p0_val = min(data_gen_selected);
p0_val = reshape(table2array(p0_val), [3,1])
p100_val = max(data_gen_selected);
p100_val = reshape(table2array(p100_val), [3,1])
skim_variable = [{'price'; 'mileage'; 'action'}];
skim_variable
% 
% Create a table to store the results
data_gen_table = table( mean_val, sd_val, p0_val, p100_val, ...
                     'VariableNames', {'mean', 'sd', 'p0', 'p100'}, 'RowNames',skim_variable);
data_gen_table

table2array(data_gen_table('price',:))

% figure;
% 
% histogram(data_gen_result.price, 6);
% 
% xlabel('price');
% ylabel('count');
% 
% 
% 
% figure;
% 
% histogram(data_gen_result.mileage, 6);
% xlabel('mileage');
% ylabel('count');

% grouped_data = varfun(@(x) {numel(x), sum(x)}, data_gen_selected, 'InputVariables', 'action', 'GroupingVariables', 'mileage');
% grouped_data.Properties.VariableNames{'GroupCount'} = 'num_state';
% grouped_data.Properties.VariableNames{'Fun_action'} = 'sum_action';
% grouped_data
% grouped_data.sum_action = cell2mat(grouped_data.sum_action(:,1))+cell2mat(grouped_data.sum_action(:,2))
% 
% grouped_data
% grouped_data.prob_buy = grouped_data.sum_action ./ grouped_data.num_state;
% grouped_data.prob_buy
% bar(grouped_data.mileage, grouped_data.prob_buy);
% xlabel('mileage');
% ylabel('prob\_buy');
% 
% grouped_data = varfun(@(x) {numel(x), sum(x)}, data_gen_selected, 'InputVariables', 'action', 'GroupingVariables', 'price');
% grouped_data.Properties.VariableNames{'GroupCount'} = 'num_state';
% grouped_data.Properties.VariableNames{'Fun_action'} = 'sum_action';
% grouped_data
% grouped_data.sum_action = cell2mat(grouped_data.sum_action(:,1))+cell2mat(grouped_data.sum_action(:,2))
% 
% grouped_data
% grouped_data.prob_buy = grouped_data.sum_action ./ grouped_data.num_state;
% grouped_data.prob_buy
% bar(grouped_data.price, grouped_data.prob_buy);
% xlabel('price');
% ylabel('prob\_buy');


grouped_data = varfun(@(x) {numel(x), sum(x)}, data_gen_result, 'InputVariables', 'action', 'GroupingVariables', {'mileage', 'price'});
grouped_data 
grouped_data.Properties.VariableNames{'GroupCount'} = 'num_state';
grouped_data.Properties.VariableNames{'Fun_action'} = 'sum_action';

grouped_data.sum_action = cell2mat(grouped_data.sum_action(:,1))+cell2mat(grouped_data.sum_action(:,2))
prob_buy_obs_mat = reshape( grouped_data.num_state./ grouped_data.sum_action, [num_price_states, num_mileage_states]);


% mileage_states = [0 20 40 60 80 100]
% prob_buy_obs_mat_reshaped = reshape(prob_buy_obs_mat, [num_mileage_states, num_price_states]);
% 
% figure;
% histogram2(mileage_states, price_states, prob_buy_obs_mat_reshaped, 'DisplayStyle', 'tile', 'ShowEmptyBins', 'on');
% set(gca, 'XLim', [min(mileage_states) max(mileage_states)], 'YLim', [min(price_states) max(price_states)]);
% xlabel('Mileage');
% ylabel('Price');
% zlabel('Probability');
% title('Conditional probability of buying');
% colormap(jet);
% colorbar;
% view(-60,10);

% data_gen_result
% data_gen_groups = splitapply(@ (varargin) {table(varargin{1}, varargin{2}, varargin{3}, varargin{4}, varargin{5}, varargin{6}, varargin{7}, varargin{8}, varargin{9}, varargin{10}, varargin{11}, varargin{12},varargin{13})}, data_gen_result, findgroups(data_gen_result.consumer));
% num = length(data_gen_groups);
% for i = 1:num
%     group = data_gen_groups{i};
%     group.lag_price_id = [NaN; group.Var10(1:end-1)];
%     group.lag_mileage_id = [NaN; group.Var11(1:end-1)];
%     group.lag_action = [NaN; group.Var8(1:end-1)];
%     group.Properties.VariableNames{1} = 'consumer';
%     group.Properties.VariableNames{2} = 'period';
%     group.Properties.VariableNames{3} = 'eps_type1_not_buy';
%     group.Properties.VariableNames{4} = 'eps_type1_buy';
%     group.Properties.VariableNames{5} = 'eps_unif';
%     group.Properties.VariableNames{6} = 'eps_price_state_unif';
%     group.Properties.VariableNames{7} = 'state_id_data_gen_result';
%     group.Properties.VariableNames{8} = 'action';
%     group.Properties.VariableNames{9} = 'state_id';
%     group.Properties.VariableNames{10} = 'price_id';
%     group.Properties.VariableNames{11} = 'mileage_id';
%     group.Properties.VariableNames{12} = 'price';
%     group.Properties.VariableNames{13} = 'mileage';
%     if i == 1
%         data_consumer_result = vertcat(group);
%     else    
%         data_consumer_result = vertcat(data_consumer_result, group);
%     end
% end
% data_consumer_result

data_gen1 =  data_consumer_result;

data_gen_filtered = data_gen1(data_gen1.period ~= (num_period - num_period_obs + 1), :);

num_cond_obs_mileage = varfun(@numel, data_gen_filtered, 'InputVariables', [], ...
    'GroupingVariables', {'lag_mileage_id', 'mileage_id', 'lag_action'});
num_cond_obs_mileage.Properties.VariableNames{'GroupCount'} = 'num_cond_obs';

num_cond_obs_mileage
num = height(num_cond_obs_mileage);

num_cond_obs_mileage_result = table();
for i = 1:num
    cond_obs_mileage = num_cond_obs_mileage(i,:);

    if cond_obs_mileage.lag_action == 0 && (cond_obs_mileage.lag_mileage_id >= 1 || cond_obs_mileage.lag_mileage_id<=20) && (cond_obs_mileage.lag_mileage_id == cond_obs_mileage.mileage_id || cond_obs_mileage.lag_action == 1) && cond_obs_mileage.mileage_id ==1
        cond_obs_mileage.cond_obs_mileage = "cond_obs_mileage1";
    elseif cond_obs_mileage.lag_action == 0 && (cond_obs_mileage.lag_mileage_id >=1 || cond_obs_mileage.lag_mileage_id <=19) && (cond_obs_mileage.lag_mileage_id == cond_obs_mileage.mileage_id-1 || cond_obs_mileage.lag_action == 1) && cond_obs_mileage.mileage_id == 2
        cond_obs_mileage.cond_obs_mileage = "cond_obs_mileage2";
    elseif cond_obs_mileage.lag_action == 0 && (cond_obs_mileage.lag_mileage_id >=1 || cond_obs_mileage.lag_mileage_id <=19) && (cond_obs_mileage.lag_mileage_id == cond_obs_mileage.mileage_id-2 || cond_obs_mileage.lag_action == 1) && cond_obs_mileage.mileage_id == 3
        cond_obs_mileage.cond_obs_mileage = "cond_obs_mileage3";
    elseif cond_obs_mileage.lag_action == 0 && cond_obs_mileage.lag_mileage_id == 20 && cond_obs_mileage.mileage_id == 21
        cond_obs_mileage.cond_obs_mileage = "cond_abs_mileage4";
    else 
        cond_obs_mileage.cond_obs_mileage = "other";
    end
    
    if i == 1
        num_cond_obs_mileage_result = vertcat(cond_obs_mileage);
        
    else
        num_cond_obs_mileage_result = [num_cond_obs_mileage_result; cond_obs_mileage];
    end
end
num_cond_obs_mileage_result

filtered_data = num_cond_obs_mileage_result(num_cond_obs_mileage_result.cond_obs_mileage ~= "other", :);


grouped_data = groupsummary(filtered_data, 'cond_obs_mileage', 'sum');


% Convert the num_cond_obs column to a matrix
result_matrix = table2array(grouped_data(:, 'sum_num_cond_obs'));


kappa_est = zeros(1,2);

for i=1:4
    if i > length(result_matrix)
        result_matrix(i) = 1;
    end
    result_matrix(i)
end 
kappa_est(1) = (result_matrix(2) * (result_matrix(2) + result_matrix(3) + result_matrix(4))) / ((result_matrix(2) + result_matrix(3)) * (result_matrix(1) + result_matrix(2) + result_matrix(3) + result_matrix(4)));
kappa_est(2) = (result_matrix(3) * (result_matrix(2) + result_matrix(3) + result_matrix(4))) / ((result_matrix(2) + result_matrix(3)) * (result_matrix(1) + result_matrix(2) + result_matrix(3) + result_matrix(4)));
kappa_est = reshape(kappa_est, 2,1);
Infomat_mileage_est = zeros(2,2);

Infomat_mileage_est(1,1) = (result_matrix(1)/(1-kappa_est(1) - kappa_est(2))^2)+ (result_matrix(2)/kappa_est(1)^2) + (result_matrix(4)/(kappa_est(1)+kappa_est(2))^2);
Infomat_mileage_est(1,2) = (result_matrix(1)/(1-kappa_est(1) - kappa_est(2))^2)+ (result_matrix(4)/(kappa_est(1)+kappa_est(2))^2);
Infomat_mileage_est(2,1) = Infomat_mileage_est(1,2);
Infomat_mileage_est(2,2) = (result_matrix(1)/(1-kappa_est(1) - kappa_est(2))^2)+ (result_matrix(3)/kappa_est(2)^2) + (result_matrix(4)/(kappa_est(1)+kappa_est(2))^2);
if isnan(Infomat_mileage_est(2,2))
    Infomat_mileage_est(2,2) = 1;
end

kappa_se = sqrt(diag(inv(Infomat_mileage_est)));

table(kappa_est, kappa_se)




% 4.2 価格の遷移行列の推定

% # それぞれの確率が実現した観察の数を数える

% 1期目は推定に使えないため落とす
data_gen_filtered = data_gen1(data_gen1.period ~= (num_period - num_period_obs + 1), :);
% t期の価格、t+1期の価格ごとにグループ化して、観察数を数える
num_cond_obs_price = varfun(@numel, data_gen_filtered, 'InputVariables', [], ...
    'GroupingVariables', {'lag_price_id', 'price_id'});
num_cond_obs_price.Properties.VariableNames{'GroupCount'} = 'num_cond_obs';
% 観察数を行列（num_price_states行の正方行列）に変換
%   # price_id (t+1期の価格) を横に広げる
num_cond_obs_price = pivot(num_cond_obs_price,Columns = "price_id", Rows = "lag_price_id",DataVariable="num_cond_obs");
num_cond_obs_price = removevars(num_cond_obs_price, 'lag_price_id');
num_cond_obs_price = table2array(num_cond_obs_price);
% 最尤法の解析解により推定値を求める
lambda_est_mat = num_cond_obs_price./sum(num_cond_obs_price);
lambda_est_mat

% 最尤法の解析解により標準誤差を求める




lambda_se = []
matrix = ones(num_price_states, num_price_states);
for i = 1:num_price_states
   num_cond_obs_price_i = num_cond_obs_price(i,:);
   
   Infomat_price_est = diag(num_cond_obs_price_i([1:i-1,i+1:end]))./lambda_est_mat([1:i-1,i+1:end],[1:i-1,i+1:end]).^2+(num_cond_obs_price(i,i)./lambda_est_mat(i,i)^2).*matrix([1:i-1,i+1:end],[1:i-1,i+1:end]);

   lambda_se = [lambda_se, sqrt(diag(inv(Infomat_price_est))).'];
end


lambda_se_mat = [0, lambda_se(1:6), 0, lambda_se(7:12), 0, lambda_se(13:18), 0, lambda_se(19:24), 0, lambda_se(25:30), 0];
lambda_se_mat = reshape(lambda_se_mat, num_price_states, num_price_states).'


lambda_est_mat
lambda_est = lambda_est_mat.'
lambda_est = [lambda_est(2:7), lambda_est(9:14),lambda_est(16:21),lambda_est(23:28),lambda_est(30:35)]
lambda_est = lambda_est(:)

lambda_se_mat
lambda_se_mat = lambda_se_mat.'
lambda_es = [lambda_se_mat(2:7), lambda_se_mat(9:14),lambda_se_mat(16:21),lambda_se_mat(23:28),lambda_se_mat(30:35)]
lambda_es = lambda_es(:)


table(lambda_est, lambda_es)


% 5 パラメータの推定
% 5.1 静学的なロジットによる推定

start_time = toc;

data_gen2 = data_consumer_result;


data_gen2.Properties.VariableNames{'state_id'} = 'state_id_state_df';
data_gen2.Properties.VariableNames{'mileage_id'} = 'mileage_id_state_df';
data_gen2.Properties.VariableNames{'price'} = 'price_state_df';
data_gen2.Properties.VariableNames{'mileage'} = 'mileage_state_df';
data_gen2.Properties.VariableNames{'price_id'} = 'price_id_state_df';
% options = optimset('Display', 'iter', 'TolFun', 1e-6, 'MaxIter', 1000, 'MaxFunEvals', 10000);
% logit_stat_opt = fminsearch(@logLH_stat, theta_true, options, state_df, data_gen2);

theta_est_stat = logit_stat_opt;
theta_est_stat
end_time = toc;

disp('Runtime:')
theta_true
theta_est_stat = [0.0421    0.000687];

disp(end_time - start_time);

% options = optimset('Display', 'iter', 'TolFun', 1e-6, 'MaxIter', 1000, 'MaxFunEvals', 10000);
% hessian_state = fminsearch(@logLH_stat, theta_est_stat, options, state_df, data_gen2);

hessian = hessian_state;
square1 = [0 hessian(2)];
square2 = [hessian(2) hessian(1)]
square = [square1; square2]
theta_se_stat = sqrt(diag(inv(-square)));
theta_est_stat = reshape(theta_est_stat, 2,1);
table(theta_est_stat, theta_se_stat)


% 5.2 不動点アルゴリズムによる推定

% 推定された遷移行列を取得

trans_mat_hat = [];
kappa_est
lambda_est
gen_mileage_trans_est = gen_mileage_trans(kappa_est, num_mileage_states, num_choice);
gen_mileage_trans_est

trans_mat_hat.not_buy = kron(gen_mileage_trans_est(:,:,1), gen_price_trans(lambda_est));
trans_mat_hat.buy = kron(gen_mileage_trans_est(:,:,2), gen_price_trans(lambda_est));



start_time = toc;
logLH( theta_true,  beta, trans_mat_hat, state_df, data_gen2, num_states, num_choice, Euler_const)


% options = optimset('Display', 'iter', 'TolFun', 1e-6, 'MaxIter', 1000, 'MaxFunEvals', 10000);
% NFXP_opt = fminsearch(@logLH, theta_true, options, beta, trans_mat_hat, state_df, data_gen2, num_states, num_choice, Euler_const);


end_time = toc;
disp("Runtime:")
disp(end_time - start_time)


theta_est = NFXP_opt 
theta_est







options = optimset('Display', 'iter', 'TolFun', 1e-6, 'MaxIter', 1000, 'MaxFunEvals', 10000);
hessian_state = fminsearch(@logLH, theta_est, options, beta, trans_mat_hat, state_df, data_gen2, num_states, num_choice, Euler_const);

hessian = hessian_state;
square1 = [0 hessian(2)];
square2 = [hessian(2) hessian(1)]
square = [square1; square2]
theta_se = sqrt(diag(inv(-square)));
theta_est = reshape(theta_est, 2,1);
table(theta_est, theta_se)








