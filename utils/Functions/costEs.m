function c = costEs(vs)
% Compute the slack cost on similarly labelled pairs

global trainData Sx simPairInds difPairInds  % Global Data
global W M                    % Global Parameters
global Mp lamda1 beta% Global Constants
global Vd U L Th           % Global variables

Z1 = W{1}'*trainData.X{1};

vd = struct2vec(Vd); % Convert the non optimized variables from this function
ed = exp(vd);        % Exponentiate vs to get the slacks
es = exp(vs);        % Exponentiate vs to get the slacks (Optimized variables)

Ed = vec2struct(ed,U);
Es = vec2struct(es,L);

Mh = sqrtm(M);
d = size(M,1);
%% On Similarly Labeled Pairs----------------------------------------------

tmpInds = simPairInds{1}{1};
delZ1Z1 = Z1(:,tmpInds(:,1)) - Z1(:,tmpInds(:,2));

MhZ1Z1 = Mh*delZ1Z1;

%Compute all similar pair distances
dSimVec11 = sum(bsxfun(@times,MhZ1Z1,MhZ1Z1));

P = beta*(dSimVec11 - L{1}{1}' - Es{1}{1}');
Psm = P(P < Th); % Small values 
Plr = P(P > Th); % Large values

dSim11 = (1/beta)*sum(log(1 + exp(Psm))) + sum(Plr);

ns = length(dSimVec11);
Term1s = dSim11/ns;

%% On Differently Labeled Pairs--------------------------------------------

tmpInds = difPairInds{1}{1};
delZ1Z1 = Z1(:,tmpInds(:,1)) - Z1(:,tmpInds(:,2));

MhZ1Z1 = Mh*delZ1Z1;

%Compute all disimilar pair distances
dDifVec11 = sum(bsxfun(@times,MhZ1Z1,MhZ1Z1));

P = beta*(U{1}{1}' - Ed{1}{1}' - dDifVec11);
Psm = P(P < Th); % Small values 
Plr = P(P > Th); % Large values

dDif11 = (1/beta)*sum(log(1 + exp(Psm))) + sum(Plr);

nd = length(dDifVec11);
Term1d = dDif11/nd;

%% Cost of Slacks----------------------------------------------------------
% Similar pair slack cost
Term2s = (1/ns)*norm(Es{1}{1},2);
Term2d = (1/nd)*norm(Ed{1}{1},2);

%% M regularizer
TermM  = (1/d)*stein(M,Mp);    

%% Covariance descripency cost---------------------------------------------
Sr = 1 ; St = 2;
TermCov = (1/d)*stein(W{Sr}'*Sx{Sr}*W{Sr},W{St}'*Sx{St}*W{St});

c = (Term1s + Term1d) + (Term2s + Term2d) + lamda1*(TermCov) + TermM;

end

