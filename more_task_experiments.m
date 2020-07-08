clear all
close all
clc
addpath('utils')
addpath('data')
be=11;
%be=0;
%be=05;
% be=1;
% k=3;
 seed=167;rng(seed);
tr_vec=1:5;
%lambda_vec=[1;1e1;1e1;1e1;1e1;1e1];
lambda_vec=[1;1e1;1e1;1e1;1e1;1e1];
m=2;
beta=[1;0.9;0.5;0.2;0.8];p=100;
Moy_t(:,1)=0.2*rand(p,1);Moy_t(:,2)=-Moy_t(:,1);
N=null(Moy_t(:,1)'); N=N(:,1);
for i=1:5
    Moy1(:,m*(i-1)+1)=beta(i)*Moy_t(:,1)+N*sqrt(1-beta(i)^2);
    Moy1(:,m*(i-1)+2)=-Moy1(:,m*(i-1)+1);
    Ct(:,:,1+m*(i-1))=eye(p);Ct(:,:,2+m*(i-1))=eye(p);
end
Ct(:,:,1+m*(6-1))=eye(p);Ct(:,:,2+m*(6-1))=eye(p);
m=2;
Matrix=[7 9;8 3;2 9;5 6;3 7];
%        dataset='mnist2';
  dataset= 'synthetic';
%      dataset='mnist2';
%     lambda_vec=logspace(-2,2,10);
%    lambda=lambda_vec(1);
for hg=tr_vec
    hg
%   for tr=1:5
      
      selected_labels{hg}=Matrix(hg,:);k=hg+1;
%   end
%    dataset='synthetic';
%   dataset='office-caltech';
switch dataset
    case 'synthetic'
        p=100;
        ns=floor((1+(abs(rand(2*k,1))))*p)';
        ns(end-1)=50;ns(end)=50;
        Nb=10000;nst=[Nb*ones(1,k*m)]';
        M=[Moy1(:,1:m*(k-1)) Moy_t];
        [S,T,X_test,y_test,M,Ct] = generate_data(M,Ct,ns,nst,p,m,k);
%         [S,T,X_test,y_test,M,Ct]=generate_mvr(ns,nst,p,m,k);
%         Ct(:,:,1)=eye(p);Ct(:,:,2)=eye(p);Ct(:,:,3)=eye(p);Ct(:,:,4)=eye(p);
    case 'mnist-usps'
        G=load('MNIST_vs_USPS.mat');
        S.fts=G.X_src';S.labels=G.Y_src;
        T.fts=G.X_tar';T.labels=G.Y_tar;
        p=size(G.X_src,1);
    case 'office-caltech'
        p=100;
        tar=load('Caltech10_SURF_L10.mat');
        list{2}=load('webcam_SURF_L10.mat');
        list{3}=load('amazon_SURF_L10.mat');
        list{1}=load('dslr_SURF_L10.mat');
        D{1}=list{1}.fts;D{2}=list{2}.fts;D{3}=list{3}.fts;Ta=tar.fts;
        [D,Ta]=pca_global(D,Ta,p);
        %[D,Ta]=HOG_features(D,Ta,p);
        list{1}.fts=D{1};list{2}.fts=D{2};list{3}.fts=D{3};tar.fts=Ta;
        selected=[1,2];
        trainRatio=[1;1;1;0.1];valRatio=[0.0;0.0;0.0;0.0];testRatio=0.9;
        [S,T,X_test,X_pop,ns] = Office_extract(list,tar,selected,hg+1,trainRatio,valRatio,testRatio);
%         [M,Ct]=compute_statistics(X_pop,tr+1,m);
%         X_test=[X_test1.X_test{3} X_test1.X_test{4}];
%         X_test=[X.X{3} X.X{4}];
        nst(1)=size(X_test{1},2);nst(2)=size(X_test{2},2);
        nst
        X_test=[X_test{1} X_test{2}];
%         nst(1)=size(X.X{3},2);nst(2)=size(X.X{4},2);
        n_test=nst;
        y_test=[1*ones(nst(1),1);2*ones(nst(2),1)];
%         X=load('X.mat');
%         X_test1=load('X_test.mat');
    case 'mnist2'
        %selected_labels{3}=[5 6];
        selected_labels_target=[1 4];%k=3;
          [S,T,X_test,~,~,Xs,Xt] = HOG_MNIST_extract(selected_labels,selected_labels_target,k);
%          [S,T,X_test,~,~,Xs,Xt] = MNIST_extract(selected_labels,selected_labels_target,k);
%         X=load('X.mat');
%         X_test1=load('X_test.mat');
        ns=100*ones(1,m*k);ns(end-1)=10;ns(end)=10;
        for task=1:k-1
            S.fts{task}=[Xs{task,1}(:,1:ns(1+m*(task-1))) Xs{task,2}(:,1:ns(2+m*(task-1)))];
            X_pop{m*(task-1)+1}=Xs{task,1}(:,ns(1+m*(task-1))+1:end);X_pop{m*(task-1)+2}=Xs{task,2}(:,ns(2+m*(task-1))+1:end);
            S.labels{task}=[1*ones(ns(1+m*(task-1)),1);2*ones(ns(2+m*(task-1)),1)];
        end
        X_pop{m*(k-1)+1}=Xt{1}(:,ns(1+m*(k-1))+1:end);X_pop{m*(k-1)+2}=Xt{2}(:,ns(2+m*(k-1))+1:end);
        [M,Ct]=compute_statistics(X_pop);
        T.labels=[1*ones(ns(end-1),1);2*ones(ns(end),1)];
        T.fts=[Xt{1}(:,1:ns(end-1)) Xt{2}(:,1:ns(end))]';
%         X_test=[X_test1.X_test{3} X_test1.X_test{4}];
%         X_test=[X.X{3} X.X{4}];
        nst(1)=size(X_test{1},2);nst(2)=size(X_test{2},2);
        X_test=[X_test{1} X_test{2}];
%         nst(1)=size(X.X{3},2);nst(2)=size(X.X{4},2);
        n_test=nst;
        y_test=[1*ones(nst(1),1);2*ones(nst(2),1)];
        p=size(X_pop{1},1);
    case 'mnist'
        init_data = loadMNISTImages('train-images-idx3-ubyte');
        init_labels = loadMNISTLabels('train-labels-idx1-ubyte');
        init_test = loadMNISTImages('t10k-images-idx3-ubyte');
        init_test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
        [labels,idx_init_labels]=sort(init_labels,'ascend');
        [labels_test,idx_init_labels_test]=sort(init_test_labels,'ascend');
        data=init_data(:,idx_init_labels);
        test=init_test(:,idx_init_labels_test);
        
        init_n=length(data(1,:));test_n=length(test(1,:));
        p=length(data(:,1));
        ns=floor([1.30 2.70 1.30 1.60]*p);
        selected_labels=[1 3];
        selected_labels_target=[5 6];
        data = data/max(data(:));test = test/max(test(:));
        mean_data=mean(data,2);mean_test=mean(test,2);
        norm2_data=0;norm2_test=0;
        for i=1:init_n
            norm2_data=norm2_data+1/init_n*norm(data(:,i)-mean_data)^2;
        end
        for i=1:test_n
            norm2_test=norm2_test+1/test_n*norm(test(:,i)-mean_test)^2;
        end
        data=(data-mean_data*ones(1,size(data,2)))/sqrt(norm2_data)*sqrt(p);
        test=(test-mean_test*ones(1,size(test,2)))/sqrt(norm2_test)*sqrt(p);
        
        
        selected_data = cell(k,1);
        selected_data_target = cell(k,1);selected_test = cell(k,1);
        cascade_selected_data=[];
        j=1;
        for i=selected_labels
            selected_data{j}=data(:,labels==i);
            cascade_selected_data = [cascade_selected_data, selected_data{j}];
            j = j+1;
        end
        kc=1;
        for i=selected_labels_target
            selected_data_target{kc}=data(:,labels==i);
            selected_test{kc}=test(:,labels_test==i);
            kc=kc+1;
        end
        S.fts=[selected_data{1}(:,1:ns(1)) selected_data{2}(:,1:ns(2))]';
        S.labels=[selected_labels(1)*ones(ns(1),1);...
            selected_labels(2)*ones(ns(2),1)];
        T.fts=[selected_data_target{1}(:,1:ns(3)) selected_data_target{2}(:,1:ns(4))]';
        T.labels=[selected_labels_target(1)*ones(ns(3),1);...
            selected_labels_target(2)*ones(ns(4),1)];
        X_test=[selected_test{1} selected_test{2}];
        n_test=[size(selected_test{1},2) size(selected_test{2},2)];
        y_test=[selected_labels(1)*ones(n_test(1),1);selected_labels(2)*ones(n_test(2),1)];
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
c1(1)=1;c1(2)=2;c1(3)=3;c1(4)=4;c1(5)=5;c1(6)=6;c1(7)=7;c1(8)=8;c1(9)=9;c1(10)=10;
c2(1)=9;c2(2)=2;
%c1(1)=selected_labels(1);c1(2)=selected_labels(2);
%c2(1)=selected_labels_target(1);c2(2)=selected_labels_target(2);
yr=1:m;M11=[];M22=[];
% ns=zeros(k*m,1);
X2=[]; y1 = []; y2=[];
for jk=1:k-1
    Xsr{jk}=S.fts{jk}; ysr{jk}=S.labels{jk};
end
%Xsr = Xsr'; 
Xtar=T.fts;ytar=T.labels;
Xtar = Xtar';
for task=1:k-1
    X1{task}=[];M11{task}=[];y11{task}=[];
    for i=1:m
        X11{i,task}=Xsr{task}(:,ysr{task}==c1(i))';
        X1{task}=[X1{task} X11{i,task}'];
        M11{task}=[M11{task} mean(X11{i,task})'];
%         ns(m*(task-1)+i)=size(X11{i,task},1);
%         Ct(:,:,m*(task-1)+i)=(X11{i,task}-mean(X11{i,task},1))'*(X11{i,task}-mean(X11{i,task},1))/ns(m*(task-1)+i);
%         Ct(:,:,m*(task-1)+i)=eye(p);
%         alpha(m*(task-1)+i)=(1/p)*trace((X11{i,task}-mean(X11{i,task},1))'*(X11{i,task}-mean(X11{i,task},1))/ns(m*(task-1)+i));
        y11{task} = [y11{task} c1(i)*ones(1, ns(m*(task-1)+i))];
    end
end
y1=[];
for task=1:k-1
    y1=[y1;y11{task}'];
end
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
    M22=[M22 mean(X22{i})'];
    y2 = [y2;c1(i)*ones(ns(i+m*(k-1)), 1)];
%     Ct(:,:,i+m*(k-1))=(X22{i}-mean(X22{i},1))'*(X22{i}-mean(X22{i},1))/ns(i+m*(k-1));
    %Ct(:,:,m*(task-1)+i)=eye(p);
end
switch dataset
    case 'mnist-usps'
        X_test=[];y_test=[];
        for i=1:m
            X_test=[X_test X22{i}(subs_comp{i},:)'];
            n_test(i)=size(X22{i}(subs_comp{i},:),1);
            y_test=[y_test;i*ones(n_test(i),1)];
        end
    case 'synthetic'
        n_test=Nb*ones(m,1);
    case 'mnist'
        
end
y_test(y_test==c1(2))=-1;
%X1 = scale_data(X1,'1',p);X2 = scale_data(X2,'1',p);
for kj=1:k-1
    X1{kj}=X1{kj}/sqrt(k*p);
end
X2=X2/sqrt(k*p);
% X1 = scale_data(X1,'1',p);X2 = scale_data(X2,'1',p);
yc=[y1;y2];
yc(yc==c1(2))=-1;
% M1e=[];
% for task=1:k-1
%     M1e=[M1e M11{task}];
% end
%     M=[M1e M22];
%      Ct(:,:,1)=alpha(1)*eye(p);Ct(:,:,2)=alpha(2)*eye(p);Ct(:,:,3)=alpha(3)*eye(p);Ct(:,:,4)=alpha(4)*eye(p);
%       Ct(:,:,1)=eye(p);Ct(:,:,2)=eye(p);Ct(:,:,3)=eye(p);Ct(:,:,4)=eye(p);     
gamma=ones(k,1); gamma(end)=1e0;
lambda=lambda_vec(1:k);
% [score1,error_opt,error_th,error_emp,alpha2, b,score_th,variance_th,score_emp,var_emp,y_opt] = MLSSVRTrain_th1_centered_other(X1,X2, y1',y2,yc', gamma, lambda,M,alpha,X_test,ns','task',n_test);
[score1,error_opt,error_th,error_emp,~, ~,score_th,variance_th,score_emp,var_emp,y_opt] = MTLLSSVMTrain_binary_task(X1,X2,yc', gamma, lambda,M,Ct,X_test,ns','task',k,n_test);
[score_ls,error_emp_ls,~, ~] = Least_square_svm(X2,y2, gamma(end),M(:,1:end-1:end),X_test,ns(end-1:end),n_test);
score11=score1(1:n_test(1));
score12=score1(n_test(1)+1:sum(n_test));
 pred=zeros(length(score1),1);
 pred(score1>((mean(score11)+mean(score12))/2))=1;pred(score1<((mean(score11)+mean(score12))/2))=-1;
 yt=[ones(n_test(1),1);-ones(n_test(2),1)];
  error_em(hg)=sum(pred~=yt)/length(score1);
  if error_em(hg)>0.5
      error_em(hg)=1-error_em(hg);
  end
  
   pred_ls=zeros(length(score_ls),1);
 pred_ls(score_ls>((mean(score_ls(1:n_test(1)))+mean(score_ls(n_test(1)+1:end)))/2))=1;pred_ls(score_ls<((mean(score_ls(1:n_test(1)))+mean(score_ls(n_test(1)+1:end)))/2))=-1;
 yt=[ones(n_test(1),1);-ones(n_test(2),1)];
  error_ls_2(hg)=sum(pred_ls~=yt)/length(score_ls);
  if error_ls_2(hg)>0.5
      error_ls_2(hg)=1-error_ls_2(hg);
  end
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
[score1_opt,error_opt_opt,error_th_opt,error_emp_opt,alpha2_opt, b_opt,score_th_opt,variance_th_opt,score_emp_opt,var_emp_opt,y_opt_opt] = MTLLSSVMTrain_binary_task(X1,X2,yopt, gamma, lambda,M,Ct,X_test,ns','task',k,n_test);
score21=score1_opt(1:n_test(1));
score22=score1_opt(n_test(1)+1:sum(n_test));
pred_opt(score1_opt>((score_th_opt(end-1)+score_th_opt(end))/2))=1;pred_opt(score1_opt<((score_th_opt(end-1)+score_th_opt(end))/2))=-1;
 %yt=[ones(n_test(1),1);-ones(n_test(2),1)];
  error_em_opt(hg)=sum(pred_opt'~=yt)/length(score1_opt);
  if error_em_opt(hg)>0.5
      error_em_opt(hg)=1-error_em_opt(hg);
  end
error_empe(hg)=sum(error_emp*[n_test(1)/sum(n_test);n_test(2)/sum(n_test)]);
error_empe_opt(hg)=sum(error_emp_opt*[n_test(1)/sum(n_test);n_test(2)/sum(n_test)]);
error_the(hg)=sum(error_th(3:4)*[n_test(1)/sum(n_test);n_test(2)/sum(n_test)]);
error_the_opt(hg)=sum(error_th_opt(3:4)*[n_test(1)/sum(n_test);n_test(2)/sum(n_test)]);
error_ls(hg)=sum(error_emp_ls*[n_test(1)/sum(n_test);n_test(2)/sum(n_test)]);
end
figure
hold on
plot(tr_vec,error_em,'r*-')
plot(tr_vec,error_em_opt,'go-')
plot(tr_vec,error_the,'r*--')
plot(tr_vec,error_the_opt,'go--')
plot(tr_vec,error_ls_2,'b.-')
% fnameo1 = sprintf('score1_o_beta%d.txt', be);fnameo2 = sprintf('score2_o_beta%d.txt', be);
% save(fnameo1,'score21','-ascii')
% save(fnameo2,'score22','-ascii')
vec=zeros(2*length(tr_vec),1);
vec(1:2:end)=tr_vec;
vec(2:2:end)=error_empe;
sprintf('(%d,%d)',vec)
vec1=zeros(2*length(tr_vec),1);
vec1(1:2:end)=tr_vec;
vec1(2:2:end)=error_the;
sprintf('(%d,%d)',vec1)
vec2=zeros(2*length(tr_vec),1);
vec2(1:2:end)=tr_vec;
vec2(2:2:end)=error_empe_opt;
sprintf('(%d,%d)',vec2)
vec3=zeros(2*length(tr_vec),1);
vec3(1:2:end)=tr_vec;
vec3(2:2:end)=error_the_opt;
sprintf('(%d,%d)',vec3)
