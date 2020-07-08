function E = expStruct(V)
%Exponentiate the structure

for i = 1 : 2
    for j = 1 : 2
        E{i}{j} = exp(V{i}{j});
    end
end

end

