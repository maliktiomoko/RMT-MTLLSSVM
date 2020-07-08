function pairing(datasets)

dataParameters = dataSetupParameters();

%domainNames   = dataParameters.domainNames;
NSplits       = 20;

%domainSet = defineDomainSets(domainNames);
for dd = 1 : length(datasets)
      
    for split = 1 : NSplits
        
        clc
        disp('Pairing split data...')
        disp(['Split :' num2str(split)])
        
        %load(strcat(datasets{dd}, '_Split_',num2str(split),'.mat'))
		
		load([pwd '/utils/Functions/Data/' datasets{dd} '_Split_' num2str(split) '.mat'])
                 
        trainLabels = trainData.labels{1};
        nTrain = length(trainLabels);
        
        labelConfusion = (repmat(trainLabels,1,nTrain) == repmat(trainLabels',nTrain,1));
        
        labelConfusion = triu(labelConfusion,1);

        [rSim,cSim] = find(labelConfusion);
        simPair = [rSim cSim]; 
        
        simPairInds{1}{1} = simPair;
        
        labelConfusion = (repmat(trainLabels,1,nTrain) ~= repmat(trainLabels',nTrain,1));
        
        labelConfusion = triu(labelConfusion,1);
       
        [rDif,cDif] = find(labelConfusion);
        difPair = [rDif cDif]; 
        
        nSim       = length(rSim);
        nDif       = length(rDif);
        sampleTemp = randsample(nDif,nSim); 
        
        difPairInds{1}{1} = difPair(sampleTemp,:);
        
        simPairInds{1}{2} = [];
        simPairInds{2}{1} = [];
        simPairInds{2}{2} = [];
        
        difPairInds{1}{2} = [];
        difPairInds{2}{1} = [];
        difPairInds{2}{2} = [];
		
		save([pwd '/utils/Functions/Data/' datasets{dd} '_PairedSplit_' num2str(split) '.mat'],'trainData','testData','simPairInds','difPairInds')
        
        %save(strcat(datasets{dd},'_PairedSplit_',num2str(split),'.mat'),'trainData','testData','simPairInds','difPairInds')
        
    end

    
end


end

