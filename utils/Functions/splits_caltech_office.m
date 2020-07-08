function defineSplits()
datasets = {'Ca-W', 'W-Ca', 'Ca-A', 'A-Ca', 'W-A', 'A-D', 'D-A', 'W-D', 'Ca-D', 'D-Ca', 'A-W', 'D-W'};

seed=137
rng(seed)
for i = 1:12
data = datasets{i};
m=10;k=2;

switch data
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

Xsr=S.fts; ysr=S.labels;
Xsr = Xsr'; 

Xtar=T.fts;ytar=T.labels;
Xtar = Xtar'; 

domainSet = datasets{i};

pairedSplitsDir = [dataDir '/PairedData/' domainSet];
dataDir = [pwd '/Data'];

for gt=1:num_trials
c1(1)=1;c1(2)=2;c1(3)=3;c1(4)=4;c1(5)=5;c1(6)=6;c1(7)=7;c1(8)=8;c1(9)=9;c1(10)=10;
yr=1:m;M11=[];M22=[];ns=zeros(k*m,1);
X1=[];X2=[]; y1 = []; y2=[];
for task=1:k-1
    for i=1:m
%        X11{i,task}=Xsr{task}(:,ysr{task}==c1(i))';
		X11{i,task}=Xsr(:,ysr==c1(i))';
        X1=[X1 X11{i,task}'];
	M11=[M11 mean(X11{i,task})'];
        ns(m*(task-1)+i)=size(X11{i,task},1);
		y1 = [y1 c1(i)*ones(1, ns(m*(task-1)+i))];
        %M11=[M11 mean(X11{i,task})'];
        %C(:,:,m*(task-1)+i)=(X11{i,task}-mean(X11{i,task},1))'*(X11{i,task}-mean(X11{i,task},1))/ns(m*(task-1)+i);
    end
end
for i=1:m
    X22{i}=Xtar(:,ytar==c1(i))';
	
	ns(i+m*(k-1))=floor(size(X22{i},1)/2);
    X22{i}=Xtar(:,ytar==c1(i))';

            ns(i+m*(k-1))=floor(size(X22{i},1)/2);
            tot=1:size(X22{i},1);
            subs{i}=zeros(ns(i+m*(k-1),1));
            subs_comp{i}=zeros(size(X22{i},1)-ns(i+m*(k-1),1));
            subs{i}=randperm(size(X22{i},1),ns(i+m*(k-1)));
            subs_comp{i}=tot;subs_comp{i}(subs{i})=[];
            X2=[X2 X22{i}(subs{i},:)'];
M22=[M22 mean(X22{i}(subs{i},:))'];
%size(ytar(subs{i}))
%subs{i}
%size(subs{i})
%size(ytar)
%ytar(subs{i},:)
	y2 = [y2;c1(i)*ones(ns(i+m*(k-1)), 1)];
    nr(i)=size(X22{i},1);
    %M22=[M22 mean(X22{i}(1:ns(i+m*(k-1)),:))'];
    %C(:,:,i+m*(k-1))=(X22{i}(1:ns(i+m*(k-1)),:)-mean(X22{i}(1:ns(i+m*(k-1)),:),1))'*(X22{i}(1:ns(i+m*(k-1)),:)-mean(X22{i}(1:ns(i+m*(k-1)),:),1))/ns(i+m*(k-1));
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
  test1=[zeros((k-1)*p,sum(nr)-sum(ns(m*(k-1)+1:k*m)));test2];
  
  nst=nr-ns(m+1:m*k)';
y_true=[];
for i=1:m
    y_true=[y_true;yr(i)*ones(nst(i),1)];
end
label_test = y_true;
data_test = test2';
label_test = label_test';

trainData.X{1} = data_train.source';
trainData.X{2} = data_train.target';
trainData.labels{1} = labels_train.source';
trainData.labels{2} = labels_train.target';
testData.X{1}=[]
testData.X{2} = data_test';
testData.labels{1} = []
testData.labels{2} = label_test;

save([pwd '/Data/SplitedData/' data '/Split_' num2str(split) '.mat'],'trainData','testData')

end

end

end

