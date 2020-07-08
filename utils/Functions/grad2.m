function g = grad2(Wo)
%Gradient function for Problem 1 : Runs on paired data
global Sx                  % Global Data
global W M                 % Global Parameters
global lamda1              % Global Constants
                     
Sr = 2; 
St = 1;

Wc = W;
Wc{Sr} = Wo ; 

d = size(M,1);
 
%% Covariance descrepencey derivative
Sz{Sr} = Wc{Sr}'*Sx{Sr}*Wc{Sr};
Sz{St} = Wc{St}'*Sx{St}*Wc{St};

gradTermCov = (0.5/d)*(2*Sx{Sr}*Wc{Sr})*(inv((Sz{Sr} + Sz{St})/2) - inv(Sz{Sr}));

g = lamda1*gradTermCov;
end

