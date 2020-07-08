function [S,T,X_test,y_test,M,Ct] = generate_data(M,Ct,ns,nst,p,m,k)
%UNTITLED13 Summary of this function goes here
%   Detailed explanation goes here
%         seed=167;rng(seed);
          X1{1}=[];X2=[];X_test=[];y_test=[];
        y2=[];
        for hg=1:k-1
            X1{hg}=[];y1{hg}=[];
            for j=1:m
                X1{hg} = [X1{hg} M(:,m*(hg-1)+j)+(Ct(:,:,m*(hg-1)+j)^(1/2))*randn(p,ns(m*(hg-1)+j))];
                y1{hg}=[y1{hg};j*ones(ns(m*(hg-1)+j),1)];
            end
            S.fts{hg}=X1{hg};S.labels{hg}=y1{hg}';
        end
        for j=1:m
            X2 = [X2 M(:,m*(k-1)+j)+(Ct(:,:,m*(k-1)+j)^(1/2))*randn(p,ns(m*(k-1)+j))];
            y2=[y2;j*ones(ns(m*(k-1)+j),1)];
        end
        for j=1:m
            X_test = [X_test M(:,m*(k-1)+j)+(Ct(:,:,m*(k-1)+j)^(1/2))*randn(p,nst(j))];
            y_test=[y_test;j*ones(nst(j),1)];
        end
        X_test2=M(:,4)+randn(p,nst(1));
        T.fts=X2';
        S.labels=y1';T.labels=y2';y_test=y_test';
end
