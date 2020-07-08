function [acc,acc_opt] = RMTMTLSSVM_train(Xs,ys,Xt,yt,X_test,y_true,m)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
k=2;M11=[];M22=[];X1=[];X2=[];
Xttot=[Xt X_test];
[Xs,Xttot]=scaled_data(Xs,Xttot,'4');
Xt=Xttot(:,1:size(Xt,2));X_test=Xttot(:,size(Xt,2)+1:end);
for task=1:k-1
    for i=1:m
        X11{i,task}=Xs(:,ys==i)';
        X1=[X1 X11{i,task}'];
        M11=[M11 mean(X11{i,task})'];
        ns(m*(task-1)+i)=size(X11{i,task},1);
    end
end
for i=1:m
    X22{i}=Xt(:,yt==i)';
    X2=[X2 X22{i}'];
	ns(i+m*(k-1))=floor(size(X22{i},1));
    M22=[M22 mean(X22{i})'];
end
M=[M11 M22];
X=[X1 X2];
p=size(X,1);
X=X/sqrt(k*p);
score_th=zeros(m,2*k);score_th_opt=zeros(m,2*k);
variance_th=zeros(m,2*k);variance_th_opt=zeros(m,2*k);
%error_s=zeros(m,1);error_t=zeros(m,1);error_s_opt=zeros(m,1);error_t_opt=zeros(m,1);
y_opt=zeros(2*k,m);y_opt_opt=zeros(2*k,m);
init_order=1:k*m;
%score_test=zeros(Nb*m,m);
lambda_opt=zeros(m,1);
%
n=sum(ns);gamma=[1;1];lambda=10;
nsa=zeros(1,2*k);
nsi=reshape([zeros(1,k);reshape(ns,m,k)],1,k*m+k);
for i=1:m
    i
    for task=1:k-1
    if i~=1
        init_order(m*(task-1)+1)=m*(task-1)+i;init_order(m*(task-1)+i)=m*(task-1)+i-1;
    end
    end
    if i~=1
    init_order(m*(k-1)+1)=m*(k-1)+i;
    init_order(m*(k-1)+i)=m*(k-1)+i-1;
    end
    nsn=ns(init_order);
    Mo=M(:,init_order);
    M1s=Mo(:,1);
    M1t=Mo(:,1+m);
    ntots=sum(nsn(2:m));ntott=sum(nsn(m+2:end));
    M2s=zeros(p,1);M2t=zeros(p,1);
    for ki=2:m
        M2s=M2s+(nsn(ki)/ntots)*Mo(:,ki);
        M2t=M2t+(nsn(ki+m)/ntott)*Mo(:,ki+m);
    end
    Ms=[M1s M2s M1t M2t];
    nsa(1:2:end)=ns(1:m:end);
    nsa(2:2:end)=sum(reshape(ns,m,k))-ns(1:m:end);
    %nsa=[ns(i) sum(ns(1:m))-ns(i) ns(i+m) sum(ns(m+1:k*m))-ns(i+m)];
    Xt2=X(:,sum(nsa(1:2*(k-1)))+1:end);
    orderm=1:sum(ns(m*(k-1)+1:k*m));
    orderm1=1+sum(nsi((m+1)*(k-1)+1:(m+1)*(k-1)+i)):sum(nsi((m+1)*(k-1)+1:(m+1)*(k-1)+1+i));
    orderm2=orderm;
    orderm2(orderm1)=[];
    order2=[orderm1 orderm2];
    X2=Xt2(:,order2);
    for task=1:k-1
    Xt1{task}=X(:,1:nsa(2*(task-1)+1)+nsa(2*(task-1)+2));
    ordert=1:sum(ns(m*(task-1)+1:m*task));
    ordert1=1+sum(nsi(1+m*(task-1):i+m*(task-1))):sum(nsi(1+m*(task-1):m*(task-1)+i+1));ordert2=ordert;
    ordert2=ordert;
    ordert2(ordert1)=[];
    order1=[ordert1 ordert2];
    X1p{task}=Xt1{task}(:,order1);
    end
    J=zeros(n,m*k);
    for h=1:m*k
        J(sum(ns(1:h-1))+1:sum(ns(1:h)),h)=ones(ns(h),1);
    end
    tildey=-ones(m*k,1);tildey(1:m:end)=1;
    yc=J*tildey;
%     yc(1:nsa(1))=1;
%     yc(nsa(1)+nsa(2)+1:nsa(1)+nsa(2)+nsa(3))=1;
ne=[nsn(1);sum(nsn(2:m));nsn(m+1);sum(nsn(m+2:end))];
Jk=zeros(n,2*k);
for g=1:2*k
    Jk(sum(ne(1:g-1))+1:sum(ne(1:g)),g)=ones(ne(g),1);
end
       [score1(:,i),error_th,alpha2, b,score_th(i,:),variance_th(i,:),score_emp,var_emp,y_opt(:,i)] = MLSSVRTrain_th1_centered(X1p{1},X2, yc, gamma, lambda,Ms,X_test,ne,'task');
       error=@(x) perf_multi(X1p{1},X2, [x(1);x(2)], x(3),Ms,ne);
       init0=[0.1;1;1];
       [param_opt,error_optim]=fmincon(error,init0,[-1 0 0;0 -1 0;0 0 -1],[0;0;0]);
       gamma_opt1(i)=param_opt(1);gamma_opt2(i)=param_opt(2);lambda_opt(i)=param_opt(3);
       yopt=Jk*y_opt(:,i);
       [score1_opt(:,i),error_th_opt,alpha2_opt, b_opt,score_th_opt(i,:),variance_th_opt(i,:),score_emp_opt,var_emp_opt,y_opt_opt(:,i)] = MLSSVRTrain_th1_centered(X1p{1},X2, yopt, [gamma(1);gamma(2)], lambda,Ms,X_test,ne,'task');
end
 [maxi_f,pred]=max(real(score1),[],2);
error=sum(pred~=y_true');
[~,pred_opt]=max(real(score1_opt),[],2);
error_opti=sum(pred_opt~=y_true');
acc=1 - error/size(X_test,2);
acc_opt=1 - error_opti/size(X_test,2);
end

