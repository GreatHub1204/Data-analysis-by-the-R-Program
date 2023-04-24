function vectorize = mat_ij_test(state_id, action, mat)
    if length(state_id)~=1
        for i=1:length(state_id)
            vectorize = mat(state_id(i), action(i));
            if i==1
                matVector = vectorize;
            else
                matVector = horzcat(matVector, vectorize);
            end
        end
    else
        vectorize = mat(state_id, action+1);
        vectorize = horzcat(vectorize(:));
    end
end