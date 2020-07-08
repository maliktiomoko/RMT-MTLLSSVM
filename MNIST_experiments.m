clear all
close all
clc
 addpath("data")
 addpath("utils")
be=11;
%be=0;
%be=05;
% be=1;
%c_vec=linspace(0.01,1,10);
lambda_vec=logspace(-3,3,20);
%lambda_vec=1e4;
for hg=1:length(lambda_vec)
    hg
k=2;m=2;
   dataset='mnist2';
  %dataset='synthetic';
switch dataset
    case 'synthetic'
        p=100;
        ns=floor([1.30 2.70 2.30 0.60]*p)';
        Nb=10000;nst=[Nb*ones(1,k*m)]';
        [S,T,X_test,y_test,M,Ct]=generate_mvr_1(ns,nst,p,k,m);
        n_test=[Nb;Nb];
%         Ct(:,:,1)=eye(p);Ct(:,:,2)=eye(p);Ct(:,:,3)=eye(p);Ct(:,:,4)=eye(p);
    case 'mnist-usps'
        G=load('MNIST_vs_USPS.mat');
        S.fts=G.X_src';S.labels=G.Y_src;
        T.fts=G.X_tar';T.labels=G.Y_tar;
        p=size(G.X_src,1);
    case 'mnist2'
        selected_labels{1}=[7 9];
        selected_labels_target=[1 4];k=2;
        [S,T,X_test,~,~,Xs,Xt] = MNIST_extract(selected_labels,selected_labels_target,k);
%         X=load('X.mat');
%         X_test1=load('X_test.mat');
        ns=[100;100;10;10];
        S.fts=[Xs{1,1}(:,1:ns(1)) Xs{1,2}(:,1:ns(2))]';
        X_pop{1}=Xs{1,1}(:,ns(1)+1:end);X_pop{2}=Xs{1,2}(:,ns(2)+1:end);
        X_pop{3}=Xt{1}(:,ns(3)+1:end);X_pop{4}=Xt{2}(:,ns(4)+1:end);
        [M,Ct]=compute_statistics(X_pop);
        S.labels=[1*ones(ns(1),1);2*ones(ns(2),1)];
        T.labels=[1*ones(ns(3),1);2*ones(ns(4),1)];
        T.fts=[Xt{1}(:,1:ns(3)) Xt{2}(:,1:ns(4))]';
%         X_test=[X_test1.X_test{3} X_test1.X_test{4}];
%         X_test=[X.X{3} X.X{4}];
        nst(1)=size(X_test{1},2);nst(2)=size(X_test{2},2);
        X_test=[X_test{1} X_test{2}];
%         nst(1)=size(X.X{3},2);nst(2)=size(X.X{4},2);
        n_test=nst;
        y_test=[1*ones(nst(1),1);2*ones(nst(2),1)];
        p=size(X_pop{1},1);
        %M=M;
%         Ct(:,:,1)=eye(p);Ct(:,:,2)=eye(p);Ct(:,:,3)=eye(p);Ct(:,:,4)=eye(p);
%            [S,T,X_test,y_test]=generate_data(M,Ct,ns,nst,p,m);
    case 'mnist'
        selected_labels=[1 3];
        selected_labels_target=[7 8];
        [S,T,X_test,n_test,y_test,X,y] = MNIST_extract(selected_labels,selected_labels_target);
%         % recentering of the k classes
%         mean_selected_data=mean(cascade_selected_data,2);
%         norm2_selected_data=mean(sum(abs(cascade_selected_data-mean_selected_data*ones(1,size(cascade_selected_data,2))).^2));
%         
%         for j=1:length(selected_labels)
%             selected_data{j}=(selected_data{j}-mean_selected_data*ones(1,size(selected_data{j},2)))/sqrt(norm2_selected_data)*sqrt(p);
%         end
%         X=zeros(p,n);
%         X_test=zeros(p,n_test);
%         for i=1:k
%             %%% Random data picking
%             data = selected_data{i}(:,randperm(size(selected_data{i},2)));
%             X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=data(:,1:n*cs(i));
%             X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=data(:,n+1:n+n_test*cs(i));
%             
%             %W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)-means(i)*ones(1,cs(i)*n);
%             %W_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)-means(i)*ones(1,cs(i)*n_test);
%         end
end
c1(1)=1;c1(2)=7;c1(3)=3;c1(4)=4;c1(5)=5;c1(6)=6;c1(7)=7;c1(8)=8;c1(9)=9;c1(10)=10;
c2(1)=9;c2(2)=4;
% c1(1)=selected_labels(1);c1(2)=selected_labels(2);
% c2(1)=selected_labels_target(1);c2(2)=selected_labels_target(2);
yr=1:m;M11=[];M22=[];
% ns=zeros(k*m,1);
X1z=[];X2=[]; y1 = []; y2=[];
Xsr=S.fts; ysr=S.labels;
Xsr = Xsr'; 
Xtar=T.fts;ytar=T.labels;
Xtar = Xtar';
for task=1:k-1
    for i=1:m
        X11{i,task}=Xsr(:,ysr==i)';
        X1z=[X1z X11{i,task}'];
%         M11=[M11 mean(X11{i,task})'];
%          ns(m*(task-1)+i)=size(X11{i,task},1);
%            Ct(:,:,m*(task-1)+i)=(X11{i,task}-mean(X11{i,task},1))'*(X11{i,task}-mean(X11{i,task},1))/ns(m*(task-1)+i);
%         alpha(m*(task-1)+i)=(1/p)*trace((X11{i,task}-mean(X11{i,task},1))'*(X11{i,task}-mean(X11{i,task},1))/ns(m*(task-1)+i));
        y1 = [y1 i*ones(1, ns(m*(task-1)+i))];
    end
end
X1{1}=X1z;
for i=1:m
    X22{i}=Xtar(:,ytar==i)';
%     ns(i+m*(k-1))=floor(size(X22{i},1)/2);
%     tot=1:size(X22{i},1);
%     subs{i}=zeros(ns(i+m*(k-1)),1);
%     subs_comp{i}=zeros(size(X22{i},1)-ns(i+m*(k-1)),1);
%     subs{i}=randperm(size(X22{i},1),ns(i+m*(k-1)));
%     subs_comp{i}=tot;subs_comp{i}(subs{i})=[];
    X2=[X2 X22{i}'];
%     M22=[M22 mean(X22{i}(subs{i},:))'];
    y2 = [y2;i*ones(ns(i+m*(k-1)), 1)];
%        Ct(:,:,i+m*(k-1))=(X22{i}-mean(X22{i},1))'*(X22{i}-mean(X22{i},1))/ns(i+m*(k-1));
%      alpha(i+m*(k-1))=(1/p)*trace((X22{i}-mean(X22{i},1))'*(X22{i}-mean(X22{i},1))/ns(i+m*(k-1)));
end
y_test(y_test==c1(2))=-1;
%X1 = scale_data(X1,'1',p);X2 = scale_data(X2,'1',p);
X1{1}=X1{1}/sqrt(2*p);X2=X2/sqrt(2*p);
% X1 = scale_data(X1,'1',p);X2 = scale_data(X2,'1',p);
yc=[y1';y2];
yc(yc==c1(2))=-1;
%    M=[M11 M22];
%      Ct(:,:,1)=alpha(1)*eye(p);Ct(:,:,2)=alpha(2)*eye(p);Ct(:,:,3)=alpha(3)*eye(p);Ct(:,:,4)=alpha(4)*eye(p);
%       Ct(:,:,1)=eye(p);Ct(:,:,2)=eye(p);Ct(:,:,3)=eye(p);Ct(:,:,4)=eye(p);     
gamma=[1;1];lambda=lambda_vec(hg);
% Ct(:,:,1)=eye(p);Ct(:,:,2)=eye(p);Ct(:,:,3)=eye(p);Ct(:,:,4)=eye(p);
% lambda=c_vec(hg)./(1+2*c_vec(hg));
% gamma=[lambda/c_vec(hg);lambda/c_vec(hg)];
% [score1,error_opt,error_th,error_emp,alpha2, b,score_th,variance_th,score_emp,var_emp,y_opt] = MLSSVRTrain_th1_centered_other(X1,X2, y1',y2,yc', gamma, lambda,M,alpha,X_test,ns','task',n_test);
 [score1,error_opt,error_th,error_emp,alpha2, b,score_th,variance_th,score_emp,var_emp,y_opt] = MTLLSSVMTrain_binary(X1,X2,yc', gamma, lambda,M,Ct,X_test,ns,'task',k,n_test);
% [score,pred,error_th,alpha2, b,score_th,variance_th,score_emp,var_emp,y_opt] = MLSSVRTrain_th1_centered(X1{1},X2, yc', gamma, lambda,M,X_test,ns,'task')
score11=score1(1:n_test(1));
score12=score1(n_test(1)+1:sum(n_test));
 pred=zeros(length(score1),1);
 pred(score1>((mean(score11)+mean(score12))/2))=1;pred(score1<((mean(score11)+mean(score12))/2))=-1;
 yt=[ones(n_test(1),1);-ones(n_test(2),1)];
  error_em(hg)=sum(pred~=yt)/length(score1);
  if error_em(hg)>0.5
      error_em(hg)=1-error_em(hg);
  end
%  error_emp(1)=sum(pred1~=yt1)/size(tstX1,2);
%  error_emp(2)=sum(pred2~=yt2)/size(tstX2,2);
%  error_emp(4)=sum(pred4~=yt4)/size(tstX4,2);
% mean=score_th(3);
% fname1 = sprintf('score1_no_beta%d.txt', be);fname2 = sprintf('score2_no_beta%d.txt', be);
% save(fname1,'score11','-ascii')
% save(fname2,'score12','-ascii')
n=sum(ns);
J=zeros(n,m*k);
for h=1:m*k
    J(sum(ns(1:h-1))+1:sum(ns(1:h)),h)=ones(ns(h),1);
end
yopt=J*y_opt;
yopt1=yopt(1:sum(ns(1:2)));yopt2=yopt(sum(ns(1:2))+1:end);
% [score1_opt,error_opt_opt,error_th_opt,error_emp_opt,alpha2_opt, b_opt,score_th_opt,variance_th_opt,score_emp_opt,var_emp_opt,y_opt_opt] = MLSSVRTrain_th1_centered_other(X1,X2,y1',y2,yopt, gamma, lambda,M,alpha,X_test,ns','task',n_test);
[score1_opt,error_opt_opt,error_th_opt,error_emp_opt,alpha2_opt, b_opt,score_th_opt,variance_th_opt,score_emp_opt,var_emp_opt,y_opt_opt] = MTLLSSVMTrain_binary(X1,X2,yopt, gamma, lambda,M,Ct,X_test,ns,'task',k,n_test);
score21=score1_opt(1:n_test(1));
score22=score1_opt(n_test(1)+1:sum(n_test));
 pred_opt(score1_opt>((score_th_opt(3)+score_th_opt(4))/2))=1;pred_opt(score1_opt<((score_th_opt(3)+score_th_opt(4))/2))=-1;
 %yt=[ones(n_test(1),1);-ones(n_test(2),1)];
  error_em_opt(hg)=sum(pred_opt'~=yt)/length(score1_opt);
  if error_em_opt(hg)>0.5
      error_em_opt(hg)=1-error_em_opt(hg);
  end
error_empe(hg)=error_emp(2)+error_emp(1);
error_empe_opt(hg)=error_emp_opt(2)+error_emp_opt(1);
error_the(hg)=error_th(4)+error_th(3);
error_the_opt(hg)=error_th_opt(4)+error_th_opt(3);
end
figure
hold on
plot(log(lambda_vec),error_em,'r*-')
plot(log(lambda_vec),error_em_opt,'go-')
 plot(log(lambda_vec),error_the,'r*--')
 plot(log(lambda_vec),error_the_opt,'go--')
% fnameo1 = sprintf('score1_o_beta%d.txt', be);fnameo2 = sprintf('score2_o_beta%d.txt', be);
% save(fnameo1,'score21','-ascii')
% save(fnameo2,'score22','-ascii')
vec=zeros(2*length(lambda_vec),1);
vec(1:2:end)=(lambda_vec);
vec(2:2:end)=error_em;
sprintf('(%d,%d)',vec)
vec1=zeros(2*length(lambda_vec),1);
vec1(1:2:end)=(lambda_vec);
vec1(2:2:end)=error_em_opt;
sprintf('(%d,%d)',vec1)
% vec2=zeros(2*length(lambda_vec),1);
% vec2(1:2:end)=(lambda_vec);
% vec2(2:2:end)=error_empe_opt;
% sprintf('(%d,%d)',vec2)
% vec3=zeros(2*length(lambda_vec),1);
% vec3(1:2:end)=(lambda_vec);
% vec3(2:2:end)=error_the_opt;
% sprintf('(%d,%d)',vec3)

