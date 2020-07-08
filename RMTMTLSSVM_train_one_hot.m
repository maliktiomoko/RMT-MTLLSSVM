function [acc,acc_opt,acc_th,acc_th_opt] = RMTMTLSSVM_train_one_hot(Xs,ys,Xt,yt,X_test,y_true,m,M,Ct,nst)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
X1{1}=[];X2=[];k=2;
% Xttot=[Xt X_test];
% [Xs,Xttot]=scaled_data(Xs,Xttot,'1');
% Xt=Xttot(:,1:size(Xt,2));X_test=Xttot(:,size(Xt,2)+1:end);
for task=1:k-1
    for i=1:m
        X11{i,task}=Xs(:,ys==i)';
        ns(m*(task-1)+i)=size(X11{i,task},1);
%         nsi1(m*(task-1)+i)=floor(size(X11{i,task},1)/2);nsi2(m*(task-1)+i)=ns(m*(task-1)+i)-nsi1(m*(task-1)+i);
        X1{task}=[X1{task} X11{i,task}'];
%         M11=[M11 mean(X11{i,task})'];
%         M111=[M111 mean(X11{i,task}(1:nsi1(m*(task-1)+i),:))'];
%         M112=[M112 mean(X11{i,task}(nsi1(m*(task-1)+i)+1:end,:))'];
%         ns(m*(task-1)+i)=size(X11{i,task},1);
    end
end
for i=1:m
    X22{i}=Xt(:,yt==i)';
    X2=[X2 X22{i}'];
	ns(i+m*(k-1))=floor(size(X22{i},1));
%     nsi1(i+m*(k-1))=floor(size(X22{i},1)/2);nsi2(i+m*(k-1))=ns(i+m*(k-1))-nsi1(i+m*(k-1));
%     M22=[M22 mean(X22{i})'];
%     M221=[M221 mean(X22{i}(1:nsi1(i+m*(k-1)),:))'];
%     M222=[M222 mean(X22{i}(nsi1(i+m*(k-1))+1:end,:))'];
end
%          M=[M11 M22];
%        Mi1=[M111 M221];Mi2=[M112 M222];
X=[X1{1} X2];
p=size(X,1);
X=X/sqrt(k*p);n1=sum(ns(1:m));
 X1{1}=X(:,1:n1);X2=X(:,n1+1:end);
gamma=[1;1];lambda=1;
% yc(1:ns(1))=1;yc(ns(1)+1:sum(ns(1:2)))=-1;
% yc(sum(ns(1:2))+1:sum(ns(1:3)))=1;yc(sum(ns(1:3))+1:sum(ns(1:4)))=-1;yc=yc';
yc=-ones(size(X,2),m);
for i=1:m
    if i==1
        yc(1:ns(i),i)=1;
    else
        yc(1+sum(ns(1:i-1)):sum(ns(1:i)),i)=1;
    end
end
for i=1:m
    yc(1+sum(ns(1:i+m*(k-1)-1)):sum(ns(1:i+m*(k-1))),i)=1;
end
%[score1,y_opt] = MLSSVRTrain_th1_centered_one_hot(X1,X2, yc, gamma, lambda,M,Mi1,Mi2,X_test,ns,nsi1,nsi2,'task',m);
[score1,y_opt,proba_the,error_emp] = MLSSVRTrain_th1_centered_other_class_one_hot(X1,X2,yc, gamma, lambda,M,Ct,X_test,ns','task',k,nst);
 J=zeros(size(X,2),m*k);
for i=1:m*k
    J(sum(ns(1:i-1))+1:sum(ns(1:i)),i)=ones(ns(i),1);
end
yopt=J*y_opt;
[score1_opt,~,proba_the_opt,error_emp_opt] = MLSSVRTrain_th1_centered_other_class_one_hot(X1,X2,yopt, gamma, lambda,M,Ct,X_test,ns','task',k,nst);
[~,pred]=max(score1,[],2);
[~,pred_opt]=max(score1_opt,[],2);
error=sum(pred~=y_true');
error_opt=sum(pred_opt~=y_true');
% pred_opt=mode(score_opt,2);
% error_opti=sum(pred_opt~=y_true');
acc=1 - error/size(X_test,2)
acc_th=nst'*proba_the'./sum(nst)
acc_opt=1 - error_opt/size(X_test,2)
acc_th_opt=nst'*proba_the_opt'./sum(nst)

%acc_opt=1 - error_opti/size(X_test,2);
end

