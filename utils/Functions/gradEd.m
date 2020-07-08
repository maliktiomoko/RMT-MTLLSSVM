function g = gradEd(vd)
% Compute the slack cost on similarly labelled pairs

global trainData difPairInds  % Global Data
global W M                    % Global Parameters
global beta                   % Global Constants
global U                      % Global variables

Z1 = W{1}'*trainData.X{1};

ed = exp(vd); 
Ed = vec2struct(ed,U);

Mh = sqrtm(M);

%% Gradient of term 1
tmpInds = difPairInds{1}{1};
delZ1Z1 = Z1(:,tmpInds(:,1)) - Z1(:,tmpInds(:,2));

MhZ1Z1 = Mh*delZ1Z1;

%Compute all similar pair distances
dDifVec11 = sum(bsxfun(@times,MhZ1Z1,MhZ1Z1));

nd = length(dDifVec11);
dDifVec = dDifVec11';

gradTerm1d = (-1/nd)*(Ed{1}{1}.*exp(beta*(U{1}{1} - Ed{1}{1} - dDifVec))./(1+ exp(beta*(U{1}{1} - Ed{1}{1} - dDifVec))));

%% Gradient of term 3
gradTerm2d = (1/nd)*(Ed{1}{1}.^2)/norm(Ed{1}{1},2);

g = gradTerm1d + gradTerm2d;

G{1}{1} = g;
G{1}{2} = zeros(size(Ed{1}{2}));
G{2}{1} = zeros(size(Ed{2}{1}));
G{2}{2} = zeros(size(Ed{2}{2}));
g       = struct2vec(G);

end

