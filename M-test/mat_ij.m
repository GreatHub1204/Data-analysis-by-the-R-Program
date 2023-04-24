function matVector = mat_ij(state_id, action, mat)

    for i=1:length(state_id)
        vectorize = mat(state_id(i), action(i));
        if i==1
            matVector = vectorize;
        else
            matVector = horzcat(matVector, vectorize);
        end
    end
end