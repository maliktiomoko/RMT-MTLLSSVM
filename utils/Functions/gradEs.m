function g = gradEs(vs)
% Compute the slack cost on similarly labelled pairs

global trainData simPairInds      % Global Data
global W M                        % Global Parameters
global beta                       % Global Constants
global L                          % Global variables

Z1 = W{1}'*trainData.X{1};

es = exp(vs);
Es = vec2struct(es,L);

Mh = sqrtm(M);

%% Gradient of term 1
tmpInds = simPairInds{1}{1};
delZ1Z1 = Z1(:,tmpInds(:,1)) - Z1(:,tmpInds(:,2));

MhZ1Z1 = Mh*delZ1Z1;

%Compute all similar pair distances
dSimVec11 = sum(bsxfun(@times,MhZ1Z1,MhZ1Z1));

ns = length(dSimVec11);
dSimVec = dSimVec11';

gradTerm1s = (-1/ns)*(Es{1}{1}.*exp(beta*(dSimVec - L{1}{1} - Es{1}{1}))./(1+ exp(beta*(dSimVec - L{1}{1} - Es{1}{1}))));

%% Gradient of term 2
gradTerm2s = (1/ns)*(Es{1}{1}.^2)/norm(Es{1}{1},2);

g = gradTerm1s + gradTerm2s;

G{1}{1} = g;
G{1}{2} = zeros(size(Es{1}{2}));
G{2}{1} = zeros(size(Es{2}{1}));
G{2}{2} = zeros(size(Es{2}{2}));
g       = struct2vec(G);

end

