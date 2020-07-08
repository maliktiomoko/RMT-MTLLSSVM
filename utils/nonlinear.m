function [out5,out2] = nonlinear(y)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
global nt Ze are out Mat variance_test1
m=4;
out2= (nt'.*(Ze*y)./(sqrt(are).*(1+out)))'*Mat(:,:,m+1)*(nt'.*(Ze*y)./(sqrt(are).*(1+out)))-variance_test1(m+1);
out5=[];
end

