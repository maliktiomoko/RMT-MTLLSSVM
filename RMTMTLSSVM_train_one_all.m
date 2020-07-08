function [acc,acc_opt,acc_th,acc_th_opt] = RMTMTLSSVM_train_one_all(Xs,ys,Xt,yt,X_test,y_true,m,M,Ct,nst)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
Nb=nst(1);
k=2;M11=[];M22=[];X1{1}=[];X2=[];
% Xttot=[Xt X_test];
% [Xs,Xttot]=scaled_data(Xs,Xttot,'1');
% Xt=Xttot(:,1:size(Xt,2));X_test=Xttot(:,size(Xt,2)+1:end);
for task=1:k-1
    for i=1:m
        X11{i,task}=Xs(:,ys==i)';
        X1{task}=[X1{task} X11{i,task}'];
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
p=size(X2,1);
X1{task}=X1{task}/sqrt(k*p);
X2=X2/sqrt(k*p);
% M=[M11 M22];
X=[X1{task} X2];
%X=X/sqrt(k*p);
score_th=zeros(m*k,m);score_th_opt=zeros(m*k,m);
variance_th=zeros(m,m*k);variance_th_opt=zeros(m,m*k);
%error_s=zeros(m,1);error_t=zeros(m,1);error_s_opt=zeros(m,1);error_t_opt=zeros(m,1);
y_opt=zeros(m*k,m);y_opt_opt=zeros(m*k,m);
obj3=zeros(m*k,m);obj4=zeros(m*k,m);
init_order=1:k*m;
%score_test=zeros(Nb*m,m);
lambda_opt=zeros(m,1);
%
n=sum(ns);gamma=[0.1;1];lambda=1;
nsa=zeros(1,2*k);
nsi=reshape([zeros(1,k);reshape(ns,m,k)],1,k*m+k);
covar=zeros(m,m,m);covar_opt1=zeros(m,m,m);
covar_opt=zeros(m,m,m);score_theo1=[];
for i=1:m
    i
    for task=1:k-1
        if i~=1
            init_order(m*(task-1)+1)=m*(task-1)+i;init_order(m*(task-1)+i)=m*(task-1)+i-1;
        end
    end
     if i~=1
%         init_order1=circshift(init_order(1:m),1);
%         init_order2=circshift(init_order(m+1:end),1);
%         init_order=[init_order1 init_order2];
      init_order(m*(k-1)+1)=m*(k-1)+i;
     init_order(m*(k-1)+i)=m*(k-1)+i-1;
     end
    nsn=ns(init_order);
    Mo=M(:,init_order);Cto=Ct(:,:,init_order);
    %M1s=Mo(:,1);
    %M1t=Mo(:,1+m);
%     ntots=sum(nsn(2:m));ntott=sum(nsn(m+2:end));
%     M2s=zeros(p,1);M2t=zeros(p,1);
%     for ki=2:m
%         M2s=M2s+(nsn(ki)/ntots)*Mo(:,ki);
%         M2t=M2t+(nsn(ki+m)/ntott)*Mo(:,ki+m);
%     end
%     Ms=[M1s M2s M1t M2t];
%     nsa(1:2:end)=ns(1:m:end);
%     nsa(2:2:end)=sum(reshape(ns,m,k))-ns(1:m:end);
    %nsa=[ns(i) sum(ns(1:m))-ns(i) ns(i+m) sum(ns(m+1:k*m))-ns(i+m)];
%      Xt2=X(:,sum(ns(1:m*(k-1)))+1:end);
%      orderm=1:sum(ns(m*(k-1)+1:k*m));
%      orderm1=1+sum(nsi((m+1)*(k-1)+1:(m+1)*(k-1)+init_order(1))):sum(nsi((m+1)*(k-1)+1:(m+1)*(k-1)+1+init_order(1)));
%      orderm2=orderm;
%      orderm2(orderm1)=[];
%      order2=[orderm1 orderm2];
%      X2=Xt2(:,order2);
%      for task=1:k-1
%          Xt1{task}=X(:,1:sum(ns(1:m*(k-1))));
%          ordert=1:sum(ns(m*(task-1)+1:m*task));
%          ordert1=1+sum(nsi(1+m*(task-1):init_order(1)+m*(task-1))):sum(nsi(1+m*(task-1):m*(task-1)+init_order(1)+1));ordert2=ordert;
%          ordert2=ordert;
%          ordert2(ordert1)=[];
%          order1=[ordert1 ordert2];
%          X1p{task}=Xt1{task}(:,order1);
%      end
    J=zeros(n,m*k);
    for h=1:m*k
        J(sum(ns(1:h-1))+1:sum(ns(1:h)),h)=ones(ns(h),1);
    end
    tildey=-ones(m*k,1);
    %tildey(1:m:end)=1;
    tildey(i)=1;tildey(i+m)=1;
    yc=J*tildey;
%     yc(1:nsa(1))=1;
%     yc(nsa(1)+nsa(2)+1:nsa(1)+nsa(2)+nsa(3))=1;
% ne=[nsn(1);sum(nsn(2:m));nsn(m+1);sum(nsn(m+2:end))];
% Jk=zeros(n,2*k);
% for g=1:2*k
%     Jk(sum(ne(1:g-1))+1:sum(ne(1:g)),g)=ones(ne(g),1);
% end
        [score1(:,i),error_opt,alpha2, b,score_th(:,i),variance_th(i,:),score_emp,var_emp,y_opt(:,i),covar(:,:,i),obj3(:,i)] = MLSSVRTrain_th1_centered_other_class(X1,X2, yc, gamma, lambda,M,Ct,X_test,ns','task',k,nst,i);
%       [score1(:,i),error_opt,alpha2, b,score_th(:,i),variance_th,score_emp,var_emp,y_opt(:,i),covar(:,:,i),obj3(:,i)] = MLSSVRTrain_th1_centered_other_class_identity(X1,X2, yc, gamma, lambda,M,X_test,nsn','task',k,nst,i);
%        error=@(x) perf_multi(X1p{1},X2, [x(1);x(2)], x(3),Ms,ne);
%        init0=[0.1;1;1];
%        [param_opt,error_optim]=fmincon(error,init0,[-1 0 0;0 -1 0;0 0 -1],[0;0;0]);
%        gamma_opt1(i)=param_opt(1);gamma_opt2(i)=param_opt(2);lambda_opt(i)=param_opt(3);
       yopt=J*y_opt(:,i);
        [score1_opt(:,i),error_opt,alpha2, b,score_th_opt(:,i),variance_th_opt(i,:),score_emp,var_emp,y_opt_opt(:,i),covar_opt1(:,:,i),obj4(:,i)] = MLSSVRTrain_th1_centered_other_class(X1,X2, yopt, gamma, lambda,M,Ct,X_test,ns','task',k,nst,i);
%        [score1_opt(:,i),error_opt,alpha2, b,score_th_opt(:,i),variance_th,score_emp,var_emp,y_opt_opt(:,i),covar_opt1(:,:,i),obj4(:,i)] = MLSSVRTrain_th1_centered_other_class_identity(X1,X2, yopt, gamma, lambda,M,X_test,nsn','task',k,nst,i);
 
end
covar_opt=zeros(m,m,m);
for j=1:m
    covar_opt(:,:,j)=Covariance_calculus(X1,X2,ns',M,Ct,gamma,lambda,obj3,j,k);
end
% for j=1:m
% score_theo1(j,:)=circshift(score_theo1(j,:),j-1);
% end
% score_theo=score_theo1;
% for j=3:m
% score_theo(j,2:end)=circshift(score_theo1(j,2:end),-j+2);
% end
for tl=1:m
covar1=covar(:,:,tl);
tl_vec=1:m;tl_vec(tl)=[];
VART=[];
for j=tl_vec
    VAR=[];
    for kv=tl_vec
        INIT=covar1([tl j],[tl kv]);
        VAR=[VAR INIT(1,1)+INIT(2,2)-INIT(1,2)-INIT(2,1)];
    end
    VART=[VART;VAR];
end
score_int3=[];
tl_vec=1:m;tl_vec(tl)=[];
for j=1:m-1
    score_int3=[score_int3 score_th(m+tl,tl)-score_th(m+tl,tl_vec(j))];
end
prob(tl)=1-mvncdf(zeros(m-1,1),inf*ones(m-1,1),score_int3',(VART+VART')/2);
end
 
 
for tl=1:m
covar1_opt=covar_opt(:,:,tl);
tl_vec_opt=1:m;tl_vec_opt(tl)=[];
VART_opt=[];
for j=tl_vec_opt
    VAR_opt=[];
    for kv=tl_vec_opt
        INIT_opt=covar1_opt([tl j],[tl kv]);
        VAR_opt=[VAR_opt INIT_opt(1,1)+INIT_opt(2,2)-INIT_opt(1,2)-INIT_opt(2,1)];
    end
    VART_opt=[VART_opt;VAR_opt];
end
score_int3_opt=[];
tl_vec_opt=1:m;tl_vec_opt(tl)=[];
for j=1:m-1
    score_int3_opt=[score_int3_opt score_th_opt(m+tl,tl)-score_th_opt(m+tl,tl_vec_opt(j))];
end
prob_opt(tl)=1-mvncdf(zeros(m-1,1),inf*ones(m-1,1),score_int3_opt',(VART_opt+VART_opt')/2);
end
 
 
 [maxi_f,pred]=max(real(score1),[],2);
error=sum(pred~=y_true');
error1(1)=sum(pred(1:Nb)~=y_true(1:Nb)')./Nb;
for i=1:m-1
    error1(i+1)=sum(pred(1+Nb*i:Nb*(i+1))~=y_true(1+Nb*i:Nb*(i+1))')./Nb;
end
error1
prob
[~,pred_opt]=max(real(score1_opt),[],2);
error_opti=sum(pred_opt~=y_true');
error1_opt(1)=sum(pred_opt(1:Nb)~=y_true(1:Nb)')./Nb;
for i=1:m-1
    error1_opt(i+1)=sum(pred_opt(1+Nb*i:Nb*(i+1))~=y_true(1+Nb*i:Nb*(i+1))')./Nb;
end
error1_opt
prob_opt
acc=1 - error/size(X_test,2);
acc_th=1-nst'*prob'./sum(nst);
acc_opt=1 - error_opti/size(X_test,2);
acc_th_opt=1-nst'*prob_opt'./sum(nst) ;
%%%%%%
vart1=variance_th(1,4);vart2=variance_th(2,5);vart3=variance_th(3,6);
vart1_opt=variance_th_opt(1,4);vart2_opt=variance_th_opt(2,5);vart3_opt=variance_th_opt(3,6);
non1c1=score_th(4:6,1)./sqrt(vart1);non1_optc1=score_th_opt(4:6,1)./sqrt(vart1_opt);
var1c1=sqrt(variance_th(1,4:6))./sqrt(vart1);var1_optc1=sqrt(variance_th_opt(1,4:6))./sqrt(vart1_opt);
non1c2=score_th(4:6,2)./sqrt(vart2);non1_optc2=score_th_opt(4:6,2)./sqrt(vart2_opt);
var1c2=sqrt(variance_th(2,4:6))./sqrt(vart2);var1_optc2=sqrt(variance_th_opt(2,4:6))./sqrt(vart2_opt);
non1c3=score_th(4:6,3)./sqrt(vart3);non1_optc3=score_th_opt(4:6,3)./sqrt(vart3_opt);
var1c3=sqrt(variance_th(3,4:6))./sqrt(vart3);var1_optc3=sqrt(variance_th_opt(3,4:6))./sqrt(vart3_opt);



end

