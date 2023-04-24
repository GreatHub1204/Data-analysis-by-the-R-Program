
% lambda_se = [];
% for i = 1:num_price_states
%    num_cond_obs_price_i = num_cond_obs_price1(i,:);
%    Infomat_price_est = diag(num_cond_obs_price_i([1:i-1,i+1:end]))./lambda_est_mat1([1:i-1,i+1:end],[1:i-1,i+1:end]).^2+(num_cond_obs_price1(i,i)./lambda_est_mat1(i,i)^2).*matrix([1:i-1,i+1:end],[1:i-1,i+1:end]);
% 
% 
%         lambda_se = [lambda_se, sqrt(diag(inv(Infomat_price_est)))'];
%         lambda_se
% end
% lambda_se_vector = [0, lambda_se(1:6), 0, lambda_se(7:12), 0, lambda_se(13:18), 0, lambda_se(19:24), 0, lambda_se(25:30), 0];
% lambda_se_vector = reshape(lambda_se_vector, num_price_states, num_price_states).';
% 
% lambda_est_mat
% lambda_est = lambda_est_mat.'
% lambda_est = [lambda_est(2:7), lambda_est(9:14),lambda_est(16:21),lambda_est(23:28),lambda_est(30:35)]
% lambda_est = lambda_est(:)
% 
% lambda_se_vector
% lambda_se_vector = lambda_se_vector.'
% lambda_es = [lambda_se_vector(2:7), lambda_se_vector(9:14),lambda_se_vector(16:21),lambda_se_vector(23:28),lambda_se_vector(30:35)]
% lambda_es = lambda_es(:)
% 
% 
% lambda_est = table(lambda_est, lambda_es);
% 
% lambda_est
options = optimoptions('fminunc', 'Algorithm', 'trust-region', 'SpecifyObjectiveGradient', true, 'HessianFcn', 'objective', 'StepTolerance', 1e-20, 'FunctionTolerance', 1e-5, 'MaxIterations', 1000, 'MaxFunctionEvaluations', 10000);
theta0 = zeros(n, 1);

[optTheta, functionVal, exitFlag] = fminunc(@(t) costFunction(X, y, lambda, t), theta0, options);



