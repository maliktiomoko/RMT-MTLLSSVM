function V = zeroStruct(E)

for i = 1 : 2
    for j = 1 : 2
        V{i}{j} = zeros(size(E{i}{j}));
    end
end

end

