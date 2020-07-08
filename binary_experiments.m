addpath('data');
addpath('utils');
X1=cell(1,1);
p=100;m=2;k=2;beta=-1;
ns=floor([1.1;1.4;0.3;0.6]*p);
nst=10000*ones(m,1);
[S,T,X_test,y_test,M,Ct] = generate_mvr(ns,nst,p,m,k,beta);
trnY=[S.labels{1};T.labels'];gamma=[1;1];lambda=10;
X1{1}=S.fts{1}./sqrt(k*p);X2=T.fts./sqrt(k*p);
[score1,error_opt,error_th,error_emp,alpha2, b,score_th,variance_th,score_emp,var_emp,y_opt,pred1,obj1] = MTLLSSVMTrain_binary(X1,X2',trnY, gamma, lambda,M,Ct,X_test,ns,'task',k,nst);
n=sum(ns);
J=zeros(n,m*k);
for h=1:m*k
    J(sum(ns(1:h-1))+1:sum(ns(1:h)),h)=ones(ns(h),1);
end
yopt=J*y_opt;
[score1,error_opt,error_th,error_emp,alpha2, b,score_th,variance_th,score_emp,var_emp,y_opt,pred1,obj1] = MTLLSSVMTrain_binary(X1,X2',yopt, gamma, lambda,M,Ct,X_test,ns,'task',k,nst);