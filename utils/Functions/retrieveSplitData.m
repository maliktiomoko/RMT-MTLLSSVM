function retrieveSplitData()
dataParameters = dataSetupParameters();

domainNames   = dataParameters.domainNames;
featureName   = dataParameters.featureName;
mainDataDir   = dataParameters.mainDataDir;
NSplits       = dataParameters.NSplits;

domainSet = defineDomainSets(domainNames);
for dd = 1 : length(domainSet)
    
    sourceDomainName = domainSet{dd}{1};
    targetDomainName = domainSet{dd}{2};
    
    load([mainDataDir '/' sourceDomainName featureName '.mat'])
    sourceLabels =  LABELS';
    sourseFTS = FTS;
    
    load([mainDataDir '/' targetDomainName featureName '.mat'])
    targetLabels =  LABELS';
    targetFTS = FTS;
    
    load([pwd '/Data/SplitDetails/' sourceDomainName '-' targetDomainName '.mat'])
    
    for split = 1 : NSplits
        
        clc
        disp('Retrieving split data...')
        disp([sourceDomainName '->' targetDomainName ])
        disp(['Split :' num2str(split)])
        
        trainData = {};
        testData  = {};
        
        % Training data on the split
        trainData.X{1}      = sourseFTS(train{split}.source,:)';
        trainData.labels{1} = sourceLabels(train{split}.source);
        
        trainData.X{2}      = [];
        trainData.labels{2} = [];
        
        % Testing data on the split
        testData.X{1}      = sourseFTS(test{split}.source,:)';
        testData.labels{1} = sourceLabels(test{split}.source);
        
        testData.X{2}      = targetFTS(test{split}.target,:)';
        testData.labels{2} = targetLabels(test{split}.target);
        
        save([pwd '/Data/SplitedData/' sourceDomainName '-' targetDomainName '/Split_' num2str(split) '.mat'],'trainData','testData','dataParameters','-v7.3')
        
    end
    
    
end

end

