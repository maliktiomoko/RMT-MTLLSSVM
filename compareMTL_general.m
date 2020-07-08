clear all;
 addpath('./utils');
 addpath('./datasets');
% AddDependencies;   %%%%% UPDATE THIS FILE TO PUT THE CORRESPONDING PATH TO LIBSVM AND DOMAIN_ECCV
% %%%% PARAMETERS
% addpath([pwd '/Functions'])
% addpath([pwd '/Functions/manopt'])
% warning('off','all')
 
 
% Choose dataset between datasets as proposed
% datasets = {'Ca-W', 'W-Ca', 'Ca-A', 'A-Ca', 'W-A', 'A-D', 'D-A', 'W-D', 'Ca-D', 'D-Ca', 'A-W', 'D-W'};
%dataset='W-Ca';
dataset='synthetic';
% dataDir = [pwd '/Data'];
 
   seed=4;
   rng(seed)
switch dataset
    case 'Ca-W'
        S=load('Caltech10_SURF_L10.mat');
        T=load('webcam_SURF_L10.mat');
    case 'W-Ca'
        S=load('webcam_SURF_L10.mat');
        T=load('Caltech10_SURF_L10.mat');
    case 'Ca-A'
        S=load('Caltech10_SURF_L10.mat');
        T=load('amazon_SURF_L10.mat');
    case 'A-Ca'
        S=load('amazon_SURF_L10.mat');
        T=load('Caltech10_SURF_L10.mat');
    case 'Ca-D'
        S=load('Caltech10_SURF_L10.mat');
        T=load('dslr_SURF_L10.mat');
    case 'D-Ca'
        S=load('dslr_SURF_L10.mat');
        T=load('Caltech10_SURF_L10.mat');
    case 'A-W'
        S=load('amazon_SURF_L10.mat');
        T=load('webcam_SURF_L10.mat');
    case 'W-A'
        S=load('webcam_SURF_L10.mat');
        T=load('amazon_SURF_L10.mat');
    
    case 'A-D'
        S=load('amazon_SURF_L10.mat');
        T=load('dslr_SURF_L10.mat');
    case 'D-A'
        S=load('dslr_SURF_L10.mat');
        T=load('amazon_SURF_L10.mat');
    case 'W-D'
        S=load('webcam_SURF_L10.mat');
        T=load('dslr_SURF_L10.mat');
    case 'D-W'
        S=load('dslr_SURF_L10.mat');
        T=load('webcam_SURF_L10.mat');
    case 'synthetic'
        p=200;m=3;k=2;beta=0;
        ns=floor((1+(abs(rand(m*k,1))))*p)';
        ns(m*(k-1)+1:end)=floor([0.1;0.9;2.4]*p)';
        Nb=10000;nst=[Nb*ones(1,m)]';
%         M=[Moy1(:,1:m*(k-1)) Moy_t];
%         [S,T,X_test,y_test,M,Ct] = generate_data(M,Ct,ns,nst,p,m,k);
         [S,T,X_test,y_test,M,Ct]=generate_mvr(ns,nst,p,m,k,beta);
        
end
 
 
 
 
% param = Config();
k=2;
Xsr{1}=S.fts{1}; ysr=S.labels;
 Xsr = Xsr{1}; 
Xtar=T.fts;ytar=T.labels;
Xtar = Xtar'; 
 
 
number_trials=1;
accuracy_mmdt = zeros(number_trials,1);
accuracy_CDLS = zeros(number_trials,1);
accuracy_ILS = zeros(number_trials,1);
accuracy_lssvm = zeros(number_trials,1);
accuracy_lssvm_opt = zeros(number_trials,1);
 
for gt=1:number_trials
gt
%%%Choose the labels for the supervision%%%
c1(1)=1;c1(2)=2;c1(3)=3;c1(4)=4;c1(5)=5;c1(6)=6;c1(7)=7;c1(8)=8;c1(9)=9;c1(10)=10;
yr=1:m;M11=[];M22=[];ns=zeros(k*m,1);
X1=[];X2=[]; y1 = []; y2=[];
for task=1:k-1
    for i=1:m
        X11{i,task}=Xsr(:,ysr{1}==c1(i))';
        X1=[X1 X11{i,task}'];
        M11=[M11 mean(X11{i,task})'];
        ns(m*(task-1)+i)=size(X11{i,task},1);
        y1 = [y1 c1(i)*ones(1, ns(m*(task-1)+i))];
    end
end
if strcmp(dataset,'synthetic')==1
for i=1:m
    X22{i}=Xtar(:,ytar==c1(i))';
    ns(i+m*(k-1))=floor(size(X22{i},1));
%     tot=1:size(X22{i},1);
%     subs{i}=zeros(ns(i+m*(k-1),1));
%     subs_comp{i}=zeros(size(X22{i},1)-ns(i+m*(k-1),1));
%     subs{i}=randperm(size(X22{i},1),ns(i+m*(k-1)));
%     subs_comp{i}=tot;subs_comp{i}(subs{i})=[];
    X2=[X2 X22{i}'];
    M22=[M22 mean(X22{i})'];
    y2 = [y2;c1(i)*ones(ns(i+m*(k-1)), 1)];
%     nr(i)=size(X22{i},1);
end
else
    for i=1:m
        X22{i}=Xtar(:,ytar==c1(i))';
        ns(i+m*(k-1))=floor(size(X22{i},1)/2);
        tot=1:size(X22{i},1);
        subs{i}=zeros(ns(i+m*(k-1),1));
        subs_comp{i}=zeros(size(X22{i},1)-ns(i+m*(k-1),1));
        subs{i}=randperm(size(X22{i},1),ns(i+m*(k-1)));
        subs_comp{i}=tot;subs_comp{i}(subs{i})=[];
        X2=[X2 X22{i}(subs{i},:)'];
        M22=[M22 mean(X22{i}(subs{i},:))'];
        y2 = [y2;c1(i)*ones(ns(i+m*(k-1)), 1)];
        nr(i)=size(X22{i},1);
    end
end
if strcmp(dataset,'synthetic')==0
    M=[M11 M22];
end
y2 = y2';
% p=size(X11{1,1},2);
% test2=[];
 
data_train.source = X1';
data_train.target = X2';
labels_train.source = y1;
labels_train.target = y2;
if strcmp(dataset,'synthetic')==0
      X_test=[];
      for i=1:m
          X_test=[X_test X22{i}(subs_comp{i},:)'];
      end
 
       nst=nr-ns(m+1:m*k)';
end
y_true=[];
for i=1:m
    y_true=[y_true;yr(i)*ones(nst(i),1)];
end
label_test = y_true;
data_test = X_test;
 
data_test_RMT = X_test;
label_test = label_test';
%%%%%%%%%%%%%%%%%%%%%%%%%% RMT METHOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 [acc_one_vs_all,acc_opt_one_vs_all,acc_th_one_vs_all,acc_opt_th_one_vs_all] = RMTMTLSSVM_train_one_vs_all(data_train.source', labels_train.source, data_train.target', labels_train.target, data_test_RMT, label_test,m,M,Ct,nst)
%   [acc_one_vs_one,acc_opt_one_vs_one,acc_th_one_vs_one,acc_opt_th_one_vs_one] = RMTMTLSSVM_train_one_vs_one(data_train.source', labels_train.source, data_train.target', labels_train.target, data_test_RMT, label_test,m,M,Ct,nst)
%  [acc_one_hot,acc_opt_one_hot,acc_th_one_hot,acc_opt_th_one_hot] = RMTMTLSSVM_train_one_hot_encoding(data_train.source', labels_train.source, data_train.target', labels_train.target, data_test_RMT, label_test,m,M,Ct,nst)
end
