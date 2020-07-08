function g = grad1(Wo)
%Gradient function for Problem 1 : Runs on paired data
global trainData Sx simPairInds difPairInds  % Global Data
global W M                                   % Global Parameters
global lamda1 beta                           % Global Constants
global Vs Vd U L                             % Global variables

Sr = 1; % Source in which the projection parameters are optimized
St = 2;

Es = expStruct(Vs);
Ed = expStruct(Vd);

Wc = W;
Wc{Sr} = Wo ; % Replace the problem 1 parameter

Mh = sqrtm(M);
d = size(M,1);

%% Similar case computation
%1) Si == S and Sj == S 
    Si = Sr;
    Sj = Sr;

    tmpInds = simPairInds{Si}{Sj};

    Xi = trainData.X{Si}(:,tmpInds(:,1));
    Xj = trainData.X{Sj}(:,tmpInds(:,2));

    Zi = Wc{Si}'*Xi;
    Zj = Wc{Sj}'*Xj;
    delZiZj = (Zi-Zj);

    MhZiZj = Mh*delZiZj;  
    dSimVecij = sum(bsxfun(@times,MhZiZj,MhZiZj));
     
    Q = exp(beta*(dSimVecij - L{Si}{Sj}' - Es{Si}{Sj}'))./(1 + exp(beta*(dSimVecij - L{Si}{Sj}' - Es{Si}{Sj}')));    
    
    gradTerm1s = 2*bsxfun(@times,Xi-Xj,Q)*delZiZj'*M;
    
    ns = length(Es{1}{1});
    gradTerm1s = gradTerm1s/ns;
%% Different case computation
%1) Si == S and Sj == S 

    Si = Sr;
    Sj = Sr;

    tmpInds = difPairInds{Si}{Sj};

    Xi = trainData.X{Si}(:,tmpInds(:,1));
    Xj = trainData.X{Sj}(:,tmpInds(:,2));

    Zi = Wc{Si}'*Xi;
    Zj = Wc{Sj}'*Xj;
    delZiZj = (Zi-Zj);

    MhZiZj = Mh*delZiZj;
    dDifVecij = sum(bsxfun(@times,MhZiZj,MhZiZj));
     
    Q = exp(beta*(U{Si}{Sj}' - Ed{Si}{Sj}'-dDifVecij))./(1 + exp(beta*(U{Si}{Sj}' - Ed{Si}{Sj}'-dDifVecij)));    
           
    gradTerm1d  = -2*bsxfun(@times,Xi-Xj,Q)*delZiZj'*M;
        
    nd = length(Ed{1}{1});
    gradTerm1d = gradTerm1d/nd;
    
%% Covariance descrepencey derivative
Sz{Sr} = Wc{Sr}'*Sx{Sr}*Wc{Sr};
Sz{St} = Wc{St}'*Sx{St}*Wc{St};

gradTermCov = (0.5/d)*(2*Sx{Sr}*Wc{Sr})*(inv((Sz{Sr} + Sz{St})/2) - inv(Sz{Sr}));

g = (gradTerm1s + gradTerm1d) + lamda1*(gradTermCov);
end

