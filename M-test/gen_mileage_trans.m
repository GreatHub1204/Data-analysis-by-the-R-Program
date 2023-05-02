function output = gen_mileage_trans(kappa, num_mileage_states, num_choice)
    kappa_1 = kappa(1);
    kappa_2 = kappa(2);

mileage_trans_mat_hat_not_buy = zeros(num_mileage_states, num_mileage_states);
for i = 1:num_mileage_states
    for j = 1:num_mileage_states
        if i == j
            mileage_trans_mat_hat_not_buy(i, j) = 1 - kappa_1 - kappa_2;
        elseif i == j - 1
            mileage_trans_mat_hat_not_buy(i, j) = kappa_1;
        elseif i == j - 2
            mileage_trans_mat_hat_not_buy(i, j) = kappa_2;
        % elseif i == j + 1
        %     mileage_trans_mat_hat_not_buy(i, j) = 0.00;
        end
    end
end

mileage_trans_mat_hat_not_buy(num_mileage_states - 1, num_mileage_states) = kappa_1 + kappa_2;
mileage_trans_mat_hat_not_buy(num_mileage_states, num_mileage_states) = 1;

mileage_trans_mat_hat_buy = ones(num_mileage_states, 1) * mileage_trans_mat_hat_not_buy(1, :);

output = reshape([mileage_trans_mat_hat_not_buy, mileage_trans_mat_hat_buy], [num_mileage_states, num_mileage_states, num_choice]);

end