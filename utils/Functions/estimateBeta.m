function beta = estimateBeta()

global trainData
global simPairInds 
global W M 

Z1 = W{1}'*trainData.X{1};
Mh = sqrtm(M);

tmpInds = simPairInds{1}{1};
delZ1Z1 = Z1(:,tmpInds(:,1)) - Z1(:,tmpInds(:,2));

MhZ1Z1 = Mh*delZ1Z1;

dSimVec = sum(bsxfun(@times,MhZ1Z1,MhZ1Z1));
beta = 1/(std(dSimVec));

end

