clear all
clc
close all
y_test=[];
y=[];
k=2;n=200;p=100;nst=10000;
X1=zeros(p,n);
Ct=toeplitz(0.7.^(0:p-1));
sigma1=ones(k,1);
for jk=1:n
    X1(:,jk)=mvnrnd(zeros(p,1),Ct);
end
X= X1/sqrt(k*p);
a1=[];
for j=1:k
%     y=[y 1*sin(ones(1,p)*X/sqrt(p))'];
     beta{j}=3*ones(1,p)/sqrt(p);
     a{j}=sigma1(j)*randn(n,1);
     y=[y (beta{j}*X)'+a{j}];
     a1=[a1;a{j}];
%  y=[y (ones(1,p)*X/sqrt(p))'];
end
X1_test=zeros(p,n);
for jk=1:nst
    X1_test(:,jk)=mvnrnd(zeros(p,1),Ct);
end
X_test= X1_test/sqrt(k*p);
%X_test = randn(p,nst);
epsilon1=[];
for j=1:k
    %y_test=[y_test 1*sin(ones(1,p)*X_test/sqrt(p))'];
%     y_test=[y_test (1*ones(1,p)*X_test/sqrt(p))'+rand(nst,1)];
    epsilon{j}=sigma1(j)*randn(nst,1);
    epsilon1=[epsilon1;epsilon{j}];
    y_test=[y_test (beta{j}*X_test)'+epsilon{j}];
end
gamma=1*ones(k,1);lambda=ones(k,1);
n_simu=10;
 %load polymer.mat
lambda_vec=logspace(-5,5,n_simu);
for i=1:n_simu
    i
    lambda=lambda_vec(i)*ones(k,1);
%     lambda=0;
    [risk(i),risk_th(i),risk_test(i),risk_test_th(i)] = MLSSVRTrain_th1_multi_output_regression(X,y, gamma, lambda,Ct,X_test',y_test,k,nst,beta,a1,sigma1,epsilon1);
end
figure
plot(log(lambda_vec),risk,'r*-')
hold on
plot(log(lambda_vec),risk_th,'go-')
figure
plot(log(lambda_vec),risk_test,'r*-')
hold on
plot(log(lambda_vec),risk_test_th,'go-')
% figure
% plot(log(lambda_vec),risk_test,'go')
% hold on
% plot(log(lambda_vec),risk_test_th,"r*")

% figure
% subplot(3,1,1)
% plot(X,y(:,1))
% hold on
% plot(X,score(1:n))
% subplot(3,1,2)
% plot(X,y(:,2))
% hold on
% plot(X,score(n+1:2*n))
% subplot(3,1,3)
% plot(X,y(:,3))
% hold on
% plot(X,score(2*n+1:3*n))