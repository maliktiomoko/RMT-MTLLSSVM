function testAcc = optNNAccuracy(testData)

global trainData
global M W

Mh = sqrtm(M);

XTrain = Mh*W{1}'*trainData.X{1};
LablesTrSVM = trainData.labels{1};

XTest = Mh*W{2}'*testData.X{2};
LablesTeSVM = testData.labels{2};
    
mdl = fitcknn(XTrain',LablesTrSVM);
[predLabelVec_Te ,~,~] = predict(mdl,XTest');

%size(predLabelVec_Te)
%size(LablesTeSVM)
testAcc  = 100*sum(predLabelVec_Te' == LablesTeSVM)/length(LablesTeSVM);

disp(['Test Accuracy :' num2str(testAcc) '%'])

end

