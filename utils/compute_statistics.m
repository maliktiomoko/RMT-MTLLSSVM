function [M,Ct]=compute_statistics(X_pop)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
p=size(X_pop{1},1);
M=zeros(p,numel(X_pop));
Ct=zeros(p,p,numel(X_pop));
for i=1:numel(X_pop)
    M(:,i)=mean(X_pop{i},2);
    Ct(:,:,i)=(X_pop{i}-mean(X_pop{i},2))*(X_pop{i}-mean(X_pop{i},2))'/size(X_pop{i},2);
end

end

