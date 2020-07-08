y_test=[];
y=[];
k=2;n=400;p=200;nst=20;
X= randn(p,n)/sqrt(k*p);
for j=1:k
%     y=[y 1*sin(ones(1,p)*X/sqrt(p))'];
     y=[y (ones(1,p)*X/sqrt(p))'+1*rand(n,1)];
%  y=[y (ones(1,p)*X/sqrt(p))'];
end
X_test = randn(p,nst);
for j=1:k
    %y_test=[y_test 1*sin(ones(1,p)*X_test/sqrt(p))'];
    y_test=[y_test (1*ones(1,p)*X_test/sqrt(p))'+rand(nst,1)];
end
gamma=1*ones(k,1);lambda=ones(k,1);Ct=eye(p);
n_simu=20;
ns=[n/2;n/2;n/2;n/2];Ct(:,:,1)=eye(p);Ct(:,:,2)=eye(p);Ct(:,:,3)=eye(p);Ct(:,:,4)=eye(p);X1{1}=X;
[score1,error_opt,error_th,error_emp,alpha2, b,score_th,variance_th,score_emp,var_emp,y_opt,pred1,obj1] = MLSSVRTrain_th1_centered_other(X1,X,y(:), gamma, lambda,zeros(p,2*k),Ct,X_test,ns,'task',2,nst)