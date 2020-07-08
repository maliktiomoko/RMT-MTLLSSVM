function V = vec2struct(v,E)

V = {};
for i = 1 : 2
    for j = i : 2
   
        n = length(E{i}{j});
        V{i}{j} = v(1:n);
        v(1:n)  = [];
        
    end
end

V{2}{1} = V{1}{2};

end

