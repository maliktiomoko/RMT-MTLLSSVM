clear all;
%datasets = 'D-W'
%AddDependencies;
%addpath('./CDLS_functions/');
%addpath('./liblinear-weights-2.30/matlab');

%%%% PARAMETERS
addpath([pwd '/Functions'])
addpath([pwd '/Functions/manopt'])
addpath('./datasets');
%addpath('/home/londonAIuser2/hafiz/multi-task_learning/code/multi-task_learning/code')
warning('off','all')


datasets = {'Ca-W', 'W-Ca', 'Ca-A', 'A-Ca', 'W-A', 'A-D', 'D-A', 'W-D', 'Ca-D', 'D-Ca', 'A-W', 'D-W'};

dataSetupParameters()

defineSplits();
pairing(datasets);

dataDir = [pwd '/Data'];

seed=137
rng(seed)

for ii = 1:12
data = datasets{ii};




param = Config();
m=10;k=2;
% Xsr{1}=Am.X_src;ysr{1}=Am.Y_src;
% Xsr{2}=Am.X_src;ysr{2}=Am.Y_src;
% Xtar=Am.X_tar;ytar=Am.Y_tar;


num_trials=1;

%%%%%%
domainSet = data;

pairedSplitsDir = [dataDir '/PairedData/' domainSet];


%%%%%%
%accuracy_mmdt = zeros(num_trials,1);
%accuracy_CDLS = zeros(num_trials,1);
accuracy_ILS = zeros(num_trials,1);
%accuracy_lssvm = zeros(num_trials,1);
%accuracy_lssvm_opt = zeros(num_trials,1);

for gt=1:num_trials
gt
%ytar
%ytar = ytar';
% XWc=Wc.fts;ywc=Wc.labels;
%%%Choose the labels for the supervision%%%
c1(1)=1;c1(2)=2;c1(3)=3;c1(4)=4;c1(5)=5;c1(6)=6;c1(7)=7;c1(8)=8;c1(9)=9;c1(10)=10;
yr=1:m;M11=[];M22=[];ns=zeros(k*m,1);
X1=[];X2=[]; y1 = []; y2=[];

%%%%%%%%%%%%%%%%%%%%%%%%%% ILS METHOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% trainData.X{1} = data_train.source';
% trainData.X{2} = data_train.target';
% trainData.labels{1} = labels_train.source';
% trainData.labels{2} = labels_train.target';
% testData.X{1}=[]
% testData.X{2} = data_test';
% testData.labels{1} = []
% testData.labels{2} = label_test;

%save([pwd datasets{i} '/Split_' num2str(gt) '.mat'],'trainData', 'testData')
global trainData Sx Xm
global simPairInds difPairInds 
global Mg W                 
global U L Vs Vd Mp lamda1 beta Th
load(strcat(data,'_PairedSplit_',num2str(gt),'.mat'),'trainData','testData','simPairInds','difPairInds')


ParameterSetup = setupParameters();



lamda1  = ParameterSetup.lamda;
Th      = 250; 

n = ParameterSetup.n; 
p = 20; 

iters       = ParameterSetup.iters;
% NSplits     = ParameterSetup.NSplits;
% domainSets  = defineDomainSets(ParameterSetup.domainNames);



        Xm{1} = mean(trainData.X{1},2);
        Xm{2} = mean(testData.X{2},2);
        
		%size(trainData.X{1})
		%size(testData.X{2})
        Pr = pca([trainData.X{1} testData.X{2}]');
        W{1} = Pr(:,1:p);
        W{2} = Pr(:,1:p);
        
%size(W{1})
        % Covariance of domains--------------------------------------------
        X{1} = bsxfun(@minus,trainData.X{1},Xm{1});
        X{2} = bsxfun(@minus,testData.X{2},Xm{2});
        
        Sx{1} = X{1}*X{1}'/(size(X{1},2) - 1);
        Sx{2} = X{2}*X{2}'/(size(X{2},2) - 1);
   
        trainData.X{1} = bsxfun(@minus,trainData.X{1},Xm{1});
        testData.X{2}  = bsxfun(@minus,testData.X{2},Xm{2});
        
%size(trainData.X{1})
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

fprintf('\n\n Mean Accuracy ILS method on %s = %6.3f +/- %6.2f\n', ...
     data, mean(accuracy_mmdt), std(accuracy_mmdt)/sqrt(num_trials));

end





