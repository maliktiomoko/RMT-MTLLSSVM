function [S,T,X_test,y_test,M,Ct] = generate_mvr_1(ns,nst,p,k,m)
%UNTITLED13 Summary of this function goes here
%   Detailed explanation goes here
          X1=[];X2=[];X_test=[];y_test=[];
         M=[];
%             seed=100;rng(seed);
    alpha=[0.3;0.9];
        for jf=1:m
            M=[M 0.6*rand(p,1)];
            Ct(:,:,jf)=toeplitz(alpha(jf).^(0:p-1));
        end
        for jf=1:m
            Ct(:,:,m+jf)=toeplitz(alpha(jf).^(0:p-1));
        end
%         M=[M -M];
%         N=null(M(:,1)'); N=N(:,1);
%         M=[M N -N];
        M=[M M];
%         for jf=1:m
%             M=[M 1.5*rand(p,1)];
%         end
         M(:,1)=[1;1;0;1;zeros(p-4,1)];
         M(:,2)=[-1;1;0;1;zeros(p-4,1)];
%          M(:,3)=[-1;1;0;-1;zeros(p-4,1)];
         orth=[ 0; 0;1;0;zeros(p-4,1)];
%         %mus2=1*rand(p,1)/sqrt(p);
%         %mut1=mus1;
%         %mut2=mus2;
%         %mut1=[ 2;0;0;0;zeros(p-4,1)];
%         %mut2=[-2;0;0;0;zeros(p-4,1)];
%         %mut1=(2/sqrt(2))*[2;-2;0;0;zeros(p-4,1)];
         beta=0.7;zeta=sqrt(1-beta^2);
%         %mut1=mus1;
         M(:,m+1)=beta*M(:,1)+zeta*orth;M(:,m+2)=beta*M(:,2)+zeta*orth;
%          M(:,6)=beta*M(:,3)+zeta*orth;
        
%         M=kron(ones(1,2),M);
        y1=[];y2=[];
        for task=1:k-1
            X1{task}=[];y1{task}=[];
            for j=1:m
                X1{task} = [X1{task} M(:,j)+Ct(:,:,j)^(1/2)*randn(p,ns(j))];
                y1{task}=[y1{task};j*ones(ns(j),1)];
            end
        end
        for j=1:m
            X2 = [X2 M(:,m+j)+Ct(:,:,j)^(1/2)*randn(p,ns(m+j))];
            y2=[y2;j*ones(ns(m+j),1)];
        end
        for j=1:m
            X_test = [X_test M(:,m+j)+Ct(:,:,j)^(1/2)*randn(p,nst(j))];
            y_test=[y_test;j*ones(nst(j),1)];
        end
        X_test2=M(:,4)+randn(p,nst(1));
        X=[X1 X2];y=[y1;y2];
        S.fts=X1{1}';T.fts=X2';
        S.labels=y1{1}';T.labels=y2';y_test=y_test';
end
