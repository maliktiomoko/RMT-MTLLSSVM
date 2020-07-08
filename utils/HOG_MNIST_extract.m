function [S,T,X_test,n_test,y_test,selected_data,selected_data_target] = HOG_MNIST_extract(selected_labels,selected_labels_target,k)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%         init_data = loadMNISTImages('train-images-idx3-ubyte');
         init_data = load('train_HOG_MNIST.mat');
        init_labels = loadMNISTLabels('train-labels-idx1-ubyte');
%         init_test = loadMNISTImages('t10k-images-idx3-ubyte');
         init_test = load('test_HOG_MNIST.mat');
        init_test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
        [labels,idx_init_labels]=sort(init_labels,'ascend');
        [labels_test,idx_init_labels_test]=sort(init_test_labels,'ascend');
        
%         data_tot=[init_data init_test];
%         numImages = size(data_tot,2);
%         cellSize = [8 8];
%         hogFeatureSize=144;
%         trainingFeatures = zeros(hogFeatureSize,numImages, 'single');
% 
% for i = 1:numImages
%     i
%     img = reshape(data_tot(:,i),sqrt(784),sqrt(784));
%     
%     % Apply pre-processing steps
%     img = imbinarize(img);
%     
%     trainingFeatures(:,i) = extractHOGFeatures(img, 'CellSize', cellSize);  
% end
% init_data=trainingFeatures(:,1:60000);
% init_test=trainingFeatures(:,60001:70000);
% save('train_HOG_MNIST.mat','init_data');
% save('test_HOG_MNIST.mat','init_test');
% % Get labels for each image.
% trainingLabels = trainingSet.Labels;
 
 
        data=init_data.init_data(:,idx_init_labels);
        test=init_test.init_test(:,idx_init_labels_test);
        
        init_n=length(data(1,:));test_n=length(test(1,:));
        p=length(data(:,1));
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
        %cascade_selected_data=[];
        for task=1:k-1
            j=1;
            cascade_selected_data{task}=[];
        for i=selected_labels{task}
            selected_data{task,j}=data(:,labels==i);
            cascade_selected_data{task} = [cascade_selected_data{task}, selected_data{task,j}];
            j = j+1;
        end
        end
        kc=1;
        for i=selected_labels_target
            selected_data_target{kc}=data(:,labels==i);
            selected_test{kc}=test(:,labels_test==i);
            kc=kc+1;
        end
%                  mean_selected_data=mean(cascade_selected_data,2);
%          norm2_selected_data=mean(sum(abs(cascade_selected_data-mean_selected_data*ones(1,size(cascade_selected_data,2))).^2));
         
%          for j=1:length(selected_labels)
%              selected_data{j}=(selected_data{j}-mean_selected_data*ones(1,size(selected_data{j},2)))/sqrt(norm2_selected_data)*sqrt(p);
%          end
        for task=1:k-1
            S.fts{task}=[selected_data{task,1} selected_data{task,2}]';
            ns(1+2*(task-1))=size(selected_data{task,1},2);
            ns(2+2*(task-1))=size(selected_data{task,2},2);
            S.labels{task}=[selected_labels{task}(1)*ones(ns(1+2*(task-1)),1);...
            selected_labels{task}(2)*ones(ns(2+2*(task-1)),1)];
        end
        ns(1+2*(k-1))=size(selected_data_target{1},2); ns(2+2*(k-1))=size(selected_data_target{2},2);
        T.fts=[selected_data_target{1} selected_data_target{2}]';
        T.labels=[selected_labels_target(1)*ones(ns(1+2*(k-1)),1);...
            selected_labels_target(2)*ones(ns(2+2*(k-1)),1)];
        X_test{1}=selected_test{1};
        X_test{2}=selected_test{2};
        n_test=[size(selected_test{1},2) size(selected_test{2},2)];
        y_test=[selected_labels_target(1)*ones(n_test(1),1);selected_labels_target(2)*ones(n_test(2),1)];
end
