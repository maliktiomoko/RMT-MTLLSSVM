function V = logStruct(E)

for i = 1 : 2
    for j = 1 : 2
        V{i}{j} = log(E{i}{j});
    end
end

end

