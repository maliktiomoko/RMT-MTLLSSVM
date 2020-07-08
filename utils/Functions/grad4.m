function g = grad4(Mo)

%Cost function for Problem 4 : Runs on paired data
global trainData simPairInds difPairInds  % Global Data
global W                      % Global Parameters
global Mp beta                % Global Constants
global Vs Vd U L              % Global variables

Z1 = W{1}'*trainData.X{1};

Es = expStruct(Vs);
Ed = expStruct(Vd);

Mh = sqrtm(Mo);
d = size(Mo,1);
%% On Similarly Labeled Pairs----------------------------------------------

tmpInds = simPairInds{1}{1};
delZ1Z1 = Z1(:,tmpInds(:,1)) - Z1(:,tmpInds(:,2));

MhZ1Z1 = Mh*delZ1Z1;

%Compute all similar pair distances
dSimVec11 = sum(bsxfun(@times,MhZ1Z1,MhZ1Z1));

Q11 = exp(beta*(dSimVec11 - L{1}{1}' - Es{1}{1}'))./(1 + exp(beta*(dSimVec11 - L{1}{1}' - Es{1}{1}')));   
QdelZ1Z1 = bsxfun(@times,delZ1Z1,Q11); 

ns = length(Es{1}{1});
gradTerm1s = (1/ns)*(QdelZ1Z1*delZ1Z1');

%% On Differently Labeled Pairs-----------------------------------------------

tmpInds = difPairInds{1}{1};
delZ1Z1 = Z1(:,tmpInds(:,1)) - Z1(:,tmpInds(:,2));

MhZ1Z1 = Mh*delZ1Z1;

%Compute all similar pair distances
dDifVec11 = sum(bsxfun(@times,MhZ1Z1,MhZ1Z1));

Q11 = -exp(beta*(U{1}{1}' - Ed{1}{1}' - dDifVec11))./(1 + exp(beta*(U{1}{1}' - Ed{1}{1}' - dDifVec11)));   

QdelZ1Z1 = bsxfun(@times,delZ1Z1,Q11); 

nd = length(Ed{1}{1});
gradTerm1d = (1/nd)*(QdelZ1Z1*delZ1Z1');

gradTermM = (0.5/d)*(inv((Mo + Mp)/2) - inv(Mo));

g = (gradTerm1s + gradTerm1d)  + gradTermM;

end

