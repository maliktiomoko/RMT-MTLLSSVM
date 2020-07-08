clear all;
%addpath([pwd '/utils/']);
addpath(genpath('./utils'));
addpath(genpath('./datasets'))
%AddDependencies;   %%%%% UPDATE THIS FILE TO PUT THE CORRESPONDING PATH TO LIBSVM AND DOMAIN_ECCV
%%%% PARAMETERS
%addpath([pwd '/utils']);
%addpath([pwd '/utils/CDLS_functions']);
%addpath([pwd '/utils/Functions']);
%addpath([pwd '/utils/DomainTransformsECCV10']);
%addpath([pwd '/utils/liblinear-weights-2.30/matlab']);
%addpath([pwd '/utils/datasets']);
%addpath([pwd '/Functions'])
%addpath([pwd '/utils/Functions/manopt'])
%addpath([pwd '/utils/datasets/surf/zscore'])
warning('off','all')


% Choose dataset between datasets as proposed
datasets = {'Ca-W', 'W-Ca', 'Ca-A', 'A-Ca', 'W-A', 'A-D', 'D-A', 'W-D', 'Ca-D', 'D-Ca', 'A-W', 'D-W'};
dataset='W-A';
features='SURF';
features='VGG';
dataDir = [pwd '/Data'];

 seed=167;
 rng(seed)
 if strcmp(features,'SURF')
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
        
end
 else
     switch dataset
    case 'Ca-W'
        %S=load('Caltech10_SURF_L10.mat');
        %T=load('webcam_SURF_L10.mat');
		S=load('caltech_VGG-FC7.mat');
		T=load('webcam_VGG-FC7.mat');
    case 'W-Ca'
        %S=load('webcam_SURF_L10.mat');
        %T=load('Caltech10_SURF_L10.mat');
		S=load('webcam_VGG-FC7.mat');
		T=load('caltech_VGG-FC7.mat');
    case 'Ca-A'
        %S=load('Caltech10_SURF_L10.mat');
        %T=load('amazon_SURF_L10.mat');
		S=load('caltech_VGG-FC7.mat');
		T=load('amazon_VGG-FC7.mat');
    case 'A-Ca'
        %S=load('amazon_SURF_L10.mat');
        %T=load('Caltech10_SURF_L10.mat');
		S=load('amazon_VGG-FC7.mat');
		T=load('caltech_VGG-FC7.mat');
    case 'Ca-D'
        %S=load('Caltech10_SURF_L10.mat');
        %T=load('dslr_SURF_L10.mat');
		S=load('caltech_VGG-FC7.mat');
		T=load('dslr_VGG-FC7.mat');
    case 'D-Ca'
        %S=load('dslr_SURF_L10.mat');
        %T=load('Caltech10_SURF_L10.mat');
		S=load('dslr_VGG-FC7.mat');
		T=load('caltech_VGG-FC7.mat');
    case 'A-W'
        %S=load('amazon_SURF_L10.mat');
        %T=load('webcam_SURF_L10.mat');
		S=load('amazon_VGG-FC7.mat');
		T=load('webcam_VGG-FC7.mat');
    case 'W-A'
        %S=load('webcam_SURF_L10.mat');
        %T=load('amazon_SURF_L10.mat');
		S=load('webcam_VGG-FC7.mat');
		T=load('amazon_VGG-FC7.mat');
	
	case 'A-D'
        %S=load('amazon_SURF_L10.mat');
        %T=load('dslr_SURF_L10.mat');
		S=load('amazon_VGG-FC7.mat');
		T=load('dslr_VGG-FC7.mat');
    case 'D-A'
        %S=load('dslr_SURF_L10.mat');
        %T=load('amazon_SURF_L10.mat');
		S=load('dslr_VGG-FC7.mat');
		T=load('amazon_VGG-FC7.mat');
    case 'W-D'
        %S=load('webcam_SURF_L10.mat');
        %T=load('dslr_SURF_L10.mat');
		S=load('webcam_VGG-FC7.mat');
		T=load('dslr_VGG-FC7.mat');
    case 'D-W'
        %S=load('dslr_SURF_L10.mat');
        %T=load('webcam_SURF_L10.mat');
		S=load('dslr_VGG-FC7.mat');
		T=load('webcam_VGG-FC7.mat');
     end
 end
     




param = Config();
m=10;k=2;
if strcmp(features,'SURF')
    Xsr=S.fts; ysr=S.labels;
    Xsr = Xsr'; 
    Xtar=T.fts;ytar=T.labels;
    Xtar = Xtar'; 
else
    Xsr = S.FTS; ysr=S.LABELS;
    Xsr = Xsr'; 
    %Xtar=T.fts;ytar=T.labels;
    Xtar = T.FTS; ytar=T.LABELS;
    Xtar=Xtar';
end


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
		X11{i,task}=Xsr(:,ysr==c1(i))';
        X1=[X1 X11{i,task}'];
        M11=[M11 mean(X11{i,task})'];
        ns(m*(task-1)+i)=size(X11{i,task},1);
		y1 = [y1 c1(i)*ones(1, ns(m*(task-1)+i))];
    end
end
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

y2 = y2';
p=size(X11{1,1},2);test2=[];

data_train.source = X1';
data_train.target = X2';
labels_train.source = y1;
labels_train.target = y2;
 for i=1:m
     test2=[test2 X22{i}(subs_comp{i},:)'];
 end
  
  nst=nr-ns(m+1:m*k)';
y_true=[];
for i=1:m
    y_true=[y_true;yr(i)*ones(nst(i),1)];
end
label_test = y_true;
data_test = test2';

data_test_RMT = test2;
label_test = label_test';
%%%%%%%%%%%%%%%%%%%%%%%%%% RMT METHOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[accuracy_lssvm(gt), accuracy_lssvm_opt(gt)] = RMTMTLSSVM_train(data_train.source', labels_train.source, data_train.target', labels_train.target, data_test_RMT, label_test,m);
clear M
%%%%%%%%%%%%%%%%%%%%%%%%% CDLS METHOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S = data_train.source;
T = data_train.target;
T_label = labels_train.target';
S_label = labels_train.source';
Ttest_labell = label_test;
Ttest = data_test;
S = S ./ repmat(sqrt(sum(S.^2,2)),1,size(S,2));
T = T ./ repmat(sqrt(sum(T.^2,2)),1,size(T,2));
Ttest = Ttest ./ repmat(sqrt(sum(Ttest.^2,2)),1,size(Ttest,2));

Data_C.T = T';
Data_C.Ttest = Ttest';
Data_C.S = S';
Data_C.T_Label = T_label;
Data_C.S_Label = S_label;
Data_C.Ttest_Label = Ttest_labell';
%%%%% Parameter Setting %%%%%
param_C.iter = 5;
param_C.scale = 0.15;
param_C.delta = 0.5; %% You can tune the portion of the weights if you like (0 < delta <= 1 )
param_C.PCA_dimension = 100; %% Make sure this dim. is smaller the source-domain dim.

%%%%% Start CDLS %%%%%
%fprintf('Transfering knowledge from Amazon images with DeCAF features to DSLR images with SURF features ...\n');
accuracy_CDLS(gt) = CDLS(Data_C,param_C);


%%%%%%%%%%%%%%%%%%%%%%%%% MMDT METHOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[model_mmdt, W2] = TrainMmdt(labels_train, data_train, param);
    %telapsed(i) = toc(tstart);
[pl, acc_mmdt, ~] = predict(label_test', ...
        [sparse(data_test), ones(length(label_test),1)], ...
        model_mmdt);
accuracy_mmdt(gt)=acc_mmdt(1);
%%%%%%%%%%%%%%%%%%%%%%%%%% ILS METHOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 dataSetupParameters()
 
 defineSplits();
 pairing(datasets);

dataDir = [pwd '/Data']; 
param = Config();
domainSet = dataset;

pairedSplitsDir = [dataDir '/PairedData/' domainSet];
global trainData Sx Xm
 global simPairInds difPairInds 
 global M W                 
 global U L Vs Vd Mp lamda1 beta Th
%load(strcat(dataset,'_PairedSplit_',num2str(gt),'.mat'),'trainData','testData','simPairInds','difPairInds')

load([pwd '/utils/Functions/Data/' dataset '_PairedSplit_' num2str(gt) '.mat'],'trainData','testData','simPairInds','difPairInds')



ParameterSetup = setupParameters();



lamda1  = ParameterSetup.lamda;
Th      = 250; 

n = ParameterSetup.n; 
p = 20; 

iters       = ParameterSetup.iters;



        Xm{1} = mean(trainData.X{1},2);
        Xm{2} = mean(testData.X{2},2);
        
		%size(trainData.X{1})
		%size(testData.X{2})
        Pr = pca([trainData.X{1} testData.X{2}]');
        W{1} = Pr(:,1:p);
        W{2} = Pr(:,1:p);
        % Covariance of domains--------------------------------------------
        X{1} = bsxfun(@minus,trainData.X{1},Xm{1});
        X{2} = bsxfun(@minus,testData.X{2},Xm{2});
        
        Sx{1} = X{1}*X{1}'/(size(X{1},2) - 1);
        Sx{2} = X{2}*X{2}'/(size(X{2},2) - 1);
   
        trainData.X{1} = bsxfun(@minus,trainData.X{1},Xm{1});
        testData.X{2}  = bsxfun(@minus,testData.X{2},Xm{2});
        Mp =  eye(p);
        M  =  eye(p);
        
        % Initialization of margins----------------------------------------

        u = 1e-2; l = 1e-2;
        initializeConstraints(u,l)
        Vs = logStruct(L);
        Vd = logStruct(U);

        u = 1; l = 1;
        initializeConstraints(u,l)

        % Define Problems--------------------------------------------------
        epsilon = 1e-5;
        options.tolgradnorm = epsilon; 
        options.maxiter = 5; 
        options.minstepsize = 1e-10;
        
        switch ParameterSetup.manifold
            case 'stiefel'
                manifoldW = stiefelfactory(n,p);
            case 'euclidean'
                manifoldW = euclideanfactory(n,p);
            otherwise
                clc
                disp('no such manifold!')
                return
        end
        
        % Problem 1
        problem1.M     = manifoldW;
        problem1.cost  = @cost1;
        problem1.egrad = @grad1;
       
        % Problem 2
        problem2.M     = manifoldW;
        problem2.cost  = @cost2;
        problem2.egrad = @grad2;
        
        % Problem 4
        manifoldM      = sympositivedefinitefactory(p);
        problem4.M     = manifoldM;
        problem4.cost  = @cost4;
        problem4.egrad = @grad4;
        
        % Optimization
        iter = 1;
        while iter < iters

            %clc
            
            beta = estimateBeta();

            W{1} = steepestdescent(problem1,W{1},options);
            W{2} = steepestdescent(problem2,W{2},options);
            M    = steepestdescent(problem4,M,options);
            slackVectorUpdate(options);
                        
            iter = iter + 1;
            
        end
        
        accuracy_ILS(gt) = optNNAccuracy(testData);


end
a_mmdt = mean(accuracy_mmdt)
std_mmdt = std(accuracy_mmdt)/sqrt(number_trials)
a_cdls = mean(accuracy_CDLS)
std_cdls = std(accuracy_CDLS)/sqrt(number_trials)
a_rmt = mean(accuracy_lssvm_opt)
std_mmdt = std(accuracy_lssvm_opt)/sqrt(number_trials)
a_lssvm = mean(accuracy_lssvm)
std_lssvm = std(accuracy_lssvm)/sqrt(number_trials)
fprintf('\n\n Mean Accuracy MMDT method on %s = %6.3f +/- %6.2f\n', ...
     dataset, mean(accuracy_mmdt), std(accuracy_mmdt)/sqrt(number_trials));
fprintf('\n\n Mean Accuracy CDLS method on %s = %6.3f +/- %6.2f\n', ...
     dataset, mean(accuracy_CDLS), std(accuracy_CDLS)/sqrt(number_trials));
 fprintf('\n\n Mean Accuracy Proposed method on %s = %6.3f +/- %6.2f\n', ...
     dataset, mean(accuracy_lssvm_opt), std(accuracy_lssvm_opt)/sqrt(number_trials));
fprintf('\n\n Mean Accuracy Non Optimized method on %s = %6.3f +/- %6.2f\n', ...
     dataset, mean(accuracy_lssvm), std(accuracy_lssvm)/sqrt(number_trials));
 fprintf('\n\n Mean Accuracy ILS method on %s = %6.3f +/- %6.2f\n', ...
     dataset, mean(accuracy_ILS), std(accuracy_ILS)/sqrt(number_trials));





