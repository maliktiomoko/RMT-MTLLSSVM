function initializeConstraints(u,l)

global simPairInds difPairInds
global U L

for i = 1 : 2
    
    for j = 1 : 2
        
       U{i}{j} = u*ones(size(simPairInds{i}{j},1),1);
	   %U(i,j) = u*ones(size(simPairInds{i}{j},1),1);
       L{i}{j} = l*ones(size(difPairInds{i}{j},1),1); 
	%   L(i,j) = l*ones(size(difPairInds{i}{j},1),1);
        
    end    
    
end

end

