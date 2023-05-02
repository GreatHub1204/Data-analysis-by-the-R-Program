% 2.2 データの生成
% 2.2.1 パラメータの設定

% 2.2.1.1 状態変数の作成
% ## パラメータの設定

% # 走行距離にかかるtheta_cと車の価格にかかるtheta_p
load carbig
% clc
% clear

theta_c = 0.004;
theta_p = 0.003;
theta_true = [theta_c, theta_p];

% 時間割引率
beta = 0.99;

% オイラー定数
Euler_const = -psi(1);

% ## Stateの作成
num_choice = 2;
% # 価格の状態変数
price_states = 2000:100:2500;

% # 走行距離の状態変数
mileage_states = 0:5:100;
% # 価格の状態変数の数
num_price_states = length(price_states);
% # 走行距離の状態変数の数
num_mileage_states = length(mileage_states);

% # 状態変数は価格と走行距離の状態変数のペア
% # 従って状態変数の数は価格の状態変数の数と走行距離の状態変数の数の積となる
num_states = num_price_states * num_mileage_states;

% # 価格、走行距離の状態変数の組み合わせ(p,m)を1つのデータフレームで表す   
state_id = (1:num_states)';
price_id = repmat(1:num_price_states, [1, num_mileage_states]);
mileage_id = repmat(1:num_mileage_states, [num_price_states, 1]);
price = repmat(price_states, [1, num_mileage_states])';
mileage = repmat(mileage_states, [num_price_states, 1]);

state_df = table(state_id, price_id(:), mileage_id(:), price(:), mileage(:));
state_df.Properties.VariableNames = {'state_id', 'price_id', 'mileage_id', 'price', 'mileage'};

tail(state_df, 3)

% 2.2.1.2 遷移行列の作成
format short

% # 走行距離の遷移行列のパラメタを設定し、遷移行列を作成する
kappa_true = [0.25, 0.05];


mileage_trans_mat_true = gen_mileage_trans(kappa_true, num_mileage_states, num_choice);
% # 走行距離の遷移行列の4行4列までを表示
mileage_trans_mat_true(1:4,1:4,1)

% # 価格の遷移行列のパラメタを設定し、遷移行列を作成する
lambda_true = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.05, 0.05, 0.1, 0.1, 0.2, 0.05, 0.05, 0.1, 0.1, 0.2];
price_trans_mat_true = gen_price_trans(lambda_true);
% # 価格の遷移行列を表示
price_trans_mat_true

% # コントロール変数毎の遷移行列を作成
trans_mat_true = [];

% # 車を購入しない場合の遷移行列
trans_mat_true.not_buy = kron(mileage_trans_mat_true(:,:,1), price_trans_mat_true);

% # 車を購入する場合の遷移行列
trans_mat_true.buy = kron(mileage_trans_mat_true(:,:,2), price_trans_mat_true);
% 
% # 定常状態での価格の分布を計算
% # 以下を満たすような price_dist_steady を求める
% # price_dist_steady %*% price_trans_mat == price_dist_steady
% 
% # 固有値/固有ベクトルを求める
% # 固有値が1となる固有ベクトルは1つだけ（1つめ）
price_trans_eigen = eig(price_trans_mat_true.');
% # 価格の定常分布を求める
[V,D] = eig(price_trans_mat_true.');

price_dist_steady = V(:,1)/sum(V(:,1));
price_dist_steady


% 2.2.1.3 効用関数の定義
% 2.2.1.4 価値関数反復法

% # EVを求める
start_time = tic;

EV_true = contraction(theta_true, beta, trans_mat_true, state_df, num_states, num_choice, Euler_const);

end_time = tic;

disp('Runtime:')
disp(end_time - start_time)
% # 選択毎の価値関数を定義する
U_true = flow_utility(theta_true, state_df);
V_CS_true = U_true + times(beta,EV_true);
V_CS_true = renamevars(V_CS_true,["EV_not_buy","EV_buy"],["V_not_buy","V_buy"]);

V_CS_true

% # state(p,m)ごとに、logitで計算される理論上の条件付き購入確率を計算
exp_V_buy = exp(V_CS_true(:, "V_buy"));
exp_V_buy = renamevars(exp_V_buy,["V_buy"],["V1"]);
exp_V_sum = sum(exp(V_CS_true),2);
exp_V_sum = renamevars(exp_V_sum,["sum"],["V1"]);

format long
prob_buy = exp_V_buy ./ exp_V_sum;
prob_buy;
prob_buy1 = table2array(prob_buy)

trans_mat_true_not_buy = reshape(prob_buy1, [num_price_states, num_mileage_states]);

trans_mat_true_not_buy

% 2.2.2 シミュレーション

% ## サンプルサイズを決める
% 
% # 1000人の消費者が存在
num_consumer = 1000;
% # 50年分の月次データを生成した後、最後の10年のみが観察できるとする
num_period = 12 * 50;

num_period_obs = 12*10;

num_obs = num_consumer * num_period;
% # 累積分布確率を持つように遷移行列を変換（行方向に足し上げる）
trans_mat_cum = [];
format short
trans_mat_cum.not_buy = cumsum(trans_mat_true.not_buy, 2);

trans_mat_cum.buy = cumsum(trans_mat_true.buy, 2);

% # 乱数を固定
rng(1)

% # 生成するデータの元となるdata.frameを作成
consumer = repmat(1:num_consumer, [num_period, 1]);
period = repmat(1:num_period,[1, num_consumer]);
eps_type1_not_buy = gevrnd(0, 1, 0, num_obs, 1);
eps_type1_buy = gevrnd(0, 1, 0, num_obs, 1);
eps_unif = unifrnd(0, 1, [num_obs 1]);
eps_price_state_unif = unifrnd(0, 1, [num_obs 1]);
state_id = zeros(num_obs, 1);
action = zeros(num_obs, 1);

data_gen = table(consumer(:), period(:),eps_type1_not_buy(:), eps_type1_buy(:),  eps_unif(:), eps_price_state_unif(:), state_id, action);
data_gen.Properties.VariableNames = {'consumer', 'period','eps_type1_not_buy', 'eps_type1_buy', 'eps_unif', 'eps_price_state_unif', 'state_id', 'action'};

data_gen_groups = splitapply(@ (consumer, period, eps_type1_not_buy, eps_type1_buy, eps_unif, eps_price_state_unif, state_id, action) {table(consumer, period, eps_type1_not_buy, eps_type1_buy, eps_unif, eps_price_state_unif, state_id, action)}, data_gen, findgroups(data_gen.consumer));
results = cell(1, numel(data_gen_groups));

for i = 1:numel(data_gen_groups)  
    group = data_gen_groups{i};
    results{i} = generate_data(group, V_CS_true, state_df, price_dist_steady, num_period, trans_mat_cum);
    data_gen_result = vertcat(results{:});
end

data_gen_result = data_gen_result(data_gen_result.period > (num_period - num_period_obs), :);
data_gen_result = outerjoin(data_gen_result, state_df, 'Keys', 'state_id');
tail(data_gen_result, 3)

clear V_CS_true trans_mat_cum;

% 3 記述統計

% data_gen_result = data_gen_result(~isnan(data_gen_result.consumer),{'consumer', 'period','eps_type1_not_buy','eps_type1_buy','eps_unif','eps_price_state_unif','state_id_data_gen_result','action','state_id_state_df','price_id','mileage_id','price', 'mileage'});
data_gen_selected = data_gen_result(:, {'price', 'mileage', 'action'});
for i=1:length(data_gen_selected.action)
    if isnan(data_gen_selected.action(i))
        data_gen_selected.action(i) = 0;
    end
end
mean_val = mean(data_gen_selected);
mean_val = reshape(table2array(mean_val), [3,1]);
sd_val = std(data_gen_selected);
sd_val = reshape(table2array(sd_val), [3,1]);
p0_val = min(data_gen_selected);
p0_val = reshape(table2array(p0_val), [3,1]);
p100_val = max(data_gen_selected);
p100_val = reshape(table2array(p100_val), [3,1]);
skim_variable = [{'price'; 'mileage'; 'action'}];


% # 生成したデータの要約統計
data_gen_table = table( mean_val, sd_val, p0_val, p100_val, ...
                     'VariableNames', {'mean', 'sd', 'p0', 'p100'}, 'RowNames',skim_variable);
data_gen_table

table2array(data_gen_table('price',:));

figure;

histogram(data_gen_selected.price, 6, FaceColor='black');

xlabel('price');
ylabel('count');



figure;

y = histogram(data_gen_selected.mileage, 21, FaceColor='black');
xlabel('mileage');
ylabel('count');


figure;

grouped_data = varfun(@(x) {numel(x), sum(x)}, data_gen_selected, 'InputVariables', 'action', 'GroupingVariables', 'mileage');
grouped_data.Properties.VariableNames{'GroupCount'} = 'num_state';
grouped_data.Properties.VariableNames{'Fun_action'} = 'sum_action';
grouped_data.sum_action = cell2mat(grouped_data.sum_action(:,1))+cell2mat(grouped_data.sum_action(:,2));
grouped_data.sum_action
grouped_data.prob_buy = (grouped_data.sum_action ./ grouped_data.num_state)-1;
grouped_data.prob_buy
bar(grouped_data.mileage, grouped_data.prob_buy);
xlabel('mileage');
ylabel('prob\_buy');


figure;

grouped_data = varfun(@(x) {numel(x), sum(x)}, data_gen_selected, 'InputVariables', 'action', 'GroupingVariables', 'price');
grouped_data.Properties.VariableNames{'GroupCount'} = 'num_state';
grouped_data.Properties.VariableNames{'Fun_action'} = 'sum_action';
grouped_data.sum_action = cell2mat(grouped_data.sum_action(:,1))+cell2mat(grouped_data.sum_action(:,2))


grouped_data.prob_buy = (grouped_data.sum_action ./ grouped_data.num_state)-1;
grouped_data.prob_buy
bar(grouped_data.price, grouped_data.prob_buy);
xlabel('price');
ylabel('prob\_buy');

figure;
grouped_data = varfun(@(x) {numel(x), sum(x)}, data_gen_result, 'InputVariables', 'action', 'GroupingVariables', {'mileage', 'price'});

grouped_data.Properties.VariableNames{'GroupCount'} = 'num_state';
grouped_data.Properties.VariableNames{'Fun_action'} = 'sum_action';

grouped_data.sum_action = cell2mat(grouped_data.sum_action(:,1))+cell2mat(grouped_data.sum_action(:,2))
prob_buy_obs_mat = reshape( -((grouped_data.num_state./ grouped_data.sum_action)-1), [num_price_states, num_mileage_states]);
prob_buy_obs_mat
prob_buy_obs_mat = transpose(prob_buy_obs_mat);



x1=price_states;
y1=mileage_states;
z=prob_buy_obs_mat; 
bar3( z (:,:,1)); 
set(gca,'XTickLabel',x1)
set(gca,'YTickLabel',y1)
xlabel('Mileage'); ylabel('Price'); zlabel('Probability');





data_gen_groups = splitapply(@ (varargin) {table(varargin{1}, varargin{2}, varargin{3}, varargin{4}, varargin{5}, varargin{6}, varargin{7}, varargin{8}, varargin{9}, varargin{10}, varargin{11}, varargin{12},varargin{13})}, data_gen_result, findgroups(data_gen_result.consumer));

num = length(data_gen_groups);
for i = 1:num
    group = data_gen_groups{i};
    group.lag_price_id = [NaN; group.Var10(1:end-1)];
    group.lag_mileage_id = [NaN; group.Var11(1:end-1)];
    group.lag_action = [NaN; group.Var8(1:end-1)];
    group.Properties.VariableNames{1} = 'consumer';
    group.Properties.VariableNames{2} = 'period';
    group.Properties.VariableNames{3} = 'eps_type1_not_buy';
    group.Properties.VariableNames{4} = 'eps_type1_buy';
    group.Properties.VariableNames{5} = 'eps_unif';
    group.Properties.VariableNames{6} = 'eps_price_state_unif';
    group.Properties.VariableNames{7} = 'state_id_data_gen_result';
    group.Properties.VariableNames{8} = 'action';
    group.Properties.VariableNames{9} = 'state_id';
    group.Properties.VariableNames{10} = 'price_id';
    group.Properties.VariableNames{11} = 'mileage_id';
    group.Properties.VariableNames{12} = 'price';
    group.Properties.VariableNames{13} = 'mileage';
    if i == 1
        data_consumer_result = vertcat(group);
    else    
        data_consumer_result = vertcat(data_consumer_result, group);
    end
end


data_gen1 =  data_consumer_result;

data_gen_filtered = data_gen1(data_gen1.period ~= (num_period - num_period_obs + 1), :);

num_cond_obs_mileage = varfun(@numel, data_gen_filtered, 'InputVariables', [], ...
    'GroupingVariables', {'lag_mileage_id', 'mileage_id', 'lag_action'});
num_cond_obs_mileage.Properties.VariableNames{'GroupCount'} = 'num_cond_obs';


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


filtered_data = num_cond_obs_mileage_result(num_cond_obs_mileage_result.cond_obs_mileage ~= "other", :);


grouped_data = groupsummary(filtered_data, 'cond_obs_mileage', 'sum');


result_matrix = table2array(grouped_data(:, 'sum_num_cond_obs'));


kappa_est = zeros(1,2);

for i=1:4
    if i > length(result_matrix)
        result_matrix(i) = 1;
    end
    result_matrix(i);
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
data_gen3 = data_consumer_result;
data_gen_filtered = data_gen3(data_gen3.period ~= (num_period - num_period_obs + 1), :);
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
for i=1:num_price_states
    for j=1:num_price_states
        if num_cond_obs_price(i,j) == 0
            num_cond_obs_price(i,j) = 1;
        end
    end
end
lambda_est_mat = num_cond_obs_price./sum(num_cond_obs_price);
lambda_est_mat

% 最尤法の解析解により標準誤差を求める




lambda_se = [];
matrixes = ones(num_price_states, num_price_states);

for i = 1:num_price_states
   num_cond_obs_price_i = num_cond_obs_price(i,:);

   Infomat_price_est = diag(num_cond_obs_price_i([1:i-1,i+1:end]))./lambda_est_mat([1:i-1,i+1:end],[1:i-1,i+1:end]).^2+(num_cond_obs_price(i,i)./lambda_est_mat(i,i)^2).*matrixes([1:i-1,i+1:end],[1:i-1,i+1:end]);

   lambda_se = [lambda_se, sqrt(diag(inv(Infomat_price_est))).'];
end


lambda_se_mat = [0, lambda_se(1:6), 0, lambda_se(7:12), 0, lambda_se(13:18), 0, lambda_se(19:24), 0, lambda_se(25:30), 0];
lambda_se_mat = reshape(lambda_se_mat, num_price_states, num_price_states).';



lambda_est = lambda_est_mat.';
lambda_est = [lambda_est(2:7), lambda_est(9:14),lambda_est(16:21),lambda_est(23:28),lambda_est(30:35)];
lambda_est = lambda_est(:);


lambda_se_mat = lambda_se_mat.'
lambda_es = [lambda_se_mat(2:7), lambda_se_mat(9:14),lambda_se_mat(16:21),lambda_se_mat(23:28),lambda_se_mat(30:35)];
lambda_es = lambda_es(:);
lambda_se_mat

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




fun =@(param) logLH_stat(param, state_df, data_gen2);

param = theta_true;
options = optimoptions(@lsqnonlin,'Algorithm','interior-point', 'Display','iter','MaxIter', 1500, 'MaxFunEvals', 10000);
x = lsqnonlin(fun, param, [], [], options);

theta_est_stat = array2table(x, 'VariableNames',{'theta_c', 'theta_p'});
disp(theta_est_stat);
theta_est_stat = table2array(theta_est_stat);
end_time = toc;

disp('Runtime:')


disp(end_time - start_time);

param = theta_est_stat;


[x,fval,exitflag,output,lambda,grad,hessian] = fmincon(fun,param);
est_divide1 = 100000;
est_divide2 = 1000000;
theta_se_stat = sqrt(diag(inv(hessian)));
theta_se_stat(1,1) = theta_se_stat(1,1)./est_divide1;
theta_se_stat(2,1) = theta_se_stat(2,1)./est_divide2;
theta_est_stat = reshape(theta_est_stat, 2,1);

table(theta_est_stat, theta_se_stat)


% 5.2 不動点アルゴリズムによる推定

% 推定された遷移行列を取得

trans_mat_hat = [];

gen_mileage_trans_est = gen_mileage_trans(kappa_est, num_mileage_states, num_choice);


trans_mat_hat.not_buy = kron(gen_mileage_trans_est(:,:,1), gen_price_trans(lambda_est));
trans_mat_hat.buy = kron(gen_mileage_trans_est(:,:,2), gen_price_trans(lambda_est));


start_time = toc;

fun =@(param) logLH(param, beta, trans_mat_hat, state_df, data_gen2, num_states, num_choice, Euler_const);

param = theta_true;
options = optimoptions(@lsqnonlin,'Algorithm','interior-point', 'Display','iter','MaxIter', 1000, 'MaxFunEvals', 10000);
x = lsqnonlin(fun, param, [], [], options);

NFXP_opt = x;

end_time = toc;
disp("Runtime:")
disp(end_time - start_time)

se_divide1 = 100000;
se_divide2 = 1000000;
theta_est = NFXP_opt;
array2table(NFXP_opt, 'VariableNames',{'theta_c', 'theta_p'})

param = theta_est;

[x,fval,exitflag,output,lambda,grad,hessian] = fmincon(fun,param);


theta_se = sqrt(diag(inv(hessian)));
theta_se(1,1) = theta_se(1,1)./se_divide1;
theta_se(2,1) = theta_se(2,1)./se_divide2;
theta_est = reshape(theta_est, 2,1);
table(theta_est, theta_se)








