function c = cost1(Wo)
%Cost function for Problem 1 : Runs on paired data

global trainData Sx simPairInds difPairInds  % Global Data
global W M                                 % Global Parameters
global Mp lamda1 beta % Global Constants
global Vs Vd U L Th                        % Global variables

Sr = 1; % Source in which the projection parameters are optimized
St = 2;

Es = expStruct(Vs);
Ed = expStruct(Vd);

Wc = W;
Wc{Sr} = Wo ; % Replace the problem 1 parameter

%size(Wc{1}')
%size(trainData.X{1})
Z1 = Wc{1}'*trainData.X{1};

d = size(M,1);
Mh = sqrtm(M);
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

% Count Similar labeled violated constraints
ns = length(dSimVec11);
Term1s = (1/ns)*dSim11;

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

% Count Similar labeled violated constraints
nd = length(dDifVec11);
Term1d = (1/nd)*dDif11;

%% Covariance descripency cost---------------------------------------------
TermCov = (1/d)*stein(Wc{Sr}'*Sx{Sr}*Wc{Sr},Wc{St}'*Sx{St}*Wc{St});

%% Cost of Slacks----------------------------------------------------------
Term2s = (1/ns)*norm(Es{1}{1}',2);
Term2d = (1/nd)*norm(Ed{1}{1}',2);

%% M regularizer
TermM  = (1/d)*stein(M,Mp);

%% Cost Sum
c = (Term1s + Term1d) + (Term2s + Term2d) + lamda1*TermCov + TermM;


end

