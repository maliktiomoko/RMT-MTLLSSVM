function dataParameters = dataSetupParameters()

dataParameters.mainDataDir   = [pwd '/Data/Features'];

dataParameters.domainNames   = {'amazon','webcam','dslr','caltech'};
dataParameters.NSplits       = 20;

dataParameters.featureName   =  '_VGG-FC6';

end

