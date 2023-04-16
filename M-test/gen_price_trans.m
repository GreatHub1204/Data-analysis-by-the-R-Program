function price_trans_mat_hat = gen_price_trans(lambda)
    lambda_11 = 1 - sum(lambda(1:5));
    lambda_22 = 1 - sum(lambda(6:10));
    lambda_33 = 1 - sum(lambda(11:15));
    lambda_44 = 1 - sum(lambda(16:20));
    lambda_55 = 1 - sum(lambda(21:25));
    lambda_66 = 1 - sum(lambda(26:30));
    price_trans_mat_hat = [lambda_11, lambda(1), lambda(2), lambda(3), lambda(4), lambda(5);                           lambda(6), lambda_22, lambda(7), lambda(8), lambda(9), lambda(10);                           lambda(11), lambda(12), lambda_33, lambda(13), lambda(14), lambda(15);                           lambda(16), lambda(17), lambda(18), lambda_44, lambda(19), lambda(20);                           lambda(21), lambda(22), lambda(23), lambda(24), lambda_55, lambda(25);                           lambda(26), lambda(27), lambda(28), lambda(29), lambda(30), lambda_66];
end