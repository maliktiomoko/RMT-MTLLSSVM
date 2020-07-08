function v = struct2vec(V)

v = [];
for i = 1 : 2
   for j = i : 2
      v = [v ; V{i}{j}]; 
   end 
end

end

