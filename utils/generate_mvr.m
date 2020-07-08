function [S,T,X_test,y_test,M,Ct] = generate_mvr(ns,nst,p,m,k,beta)
%UNTITLED13 Summary of this function goes here
%   Detailed explanation goes here
          X_test=[];y_test=[];
%           M=[];
          M=zeros(p,k*m);M_orth=zeros(p,k*m);
%          M(:,1)=2*[1;0;0;0;zeros(p-4,1)];Ct(:,:,1)=eye(p);
%          M(:,2)=2*[-1;0;0;0;zeros(p-4,1)];Ct(:,:,2)=eye(p);
%          M_orth(:,1)=2*[0;1;0;0;zeros(p-4,1)];
%          M_orth(:,2)=2*[0;-1;0;0;zeros(p-4,1)];
%           seed=167;rng(seed);
        for s=1:k-1
            for jf=1:m
                  M(m*(s-1)+jf,m*(s-1)+jf)=2;
                  M_orth(p-m*(s-1)-jf,m*(s-1)+jf)=2;
%                  M=[M 1*rand(p,1)];
%                   Ct(:,:,m*(s-1)+jf)=toeplitz(abs(rand()).^(0:p-1));
                    Ct(:,:,m*(s-1)+jf)=toeplitz(0.^(0:p-1));
            end
        end
        for jf=1:m
              M(:,m*(k-1)+jf)=beta*M(:,jf)+sqrt(1-beta^2)*M_orth(:,jf);
%              M=[M 1*rand(p,1)];
%               Ct(:,:,m*(k-1)+jf)=toeplitz((abs(rand())).^(0:p-1));
                Ct(:,:,m*(k-1)+jf)=toeplitz(0.^(0:p-1));
        end
%          M(:,1)=0.8*[1;1;0;0;zeros(p-4,1)];
%          M(:,2)=0.8*[-1;-1;0;0;zeros(p-4,1)];
%          M(:,3)=[-1;1;0;-1;zeros(p-4,1)];
%          orth=0.8*[ 0; 0;1;1;zeros(p-4,1)];
%          orth2=-0.8*[ 0; 0;1;-1;zeros(p-4,1)];
%         %mus2=1*rand(p,1)/sqrt(p);
%         %mut1=mus1;
%         %mut2=mus2;
%         %mut1=[ 2;0;0;0;zeros(p-4,1)];
%         %mut2=[-2;0;0;0;zeros(p-4,1)];
%         %mut1=(2/sqrt(2))*[2;-2;0;0;zeros(p-4,1)];
%          beta=0.75;zeta=sqrt(1-beta^2);
%         %mut1=mus1;
%          M(:,m+1)=beta*M(:,1)+zeta*orth;M(:,m+2)=beta*M(:,2)+zeta*orth2;
%          M(:,6)=beta*M(:,3)+zeta*orth;
        
%         M=kron(ones(1,2),M);
% alpha=[0.1 0.2 0.3 0.4];
%  Ct(:,:,1)=alpha(1)*eye(p);Ct(:,:,2)=alpha(2)*eye(p);
%  Ct(:,:,3)=alpha(3)*eye(p);Ct(:,:,4)=alpha(4)*eye(p);
%  Ct(:,:,1)=toeplitz(alpha(1).^(0:p-1));
%  Ct(:,:,2)=toeplitz(alpha(2).^(0:p-1));
%  Ct(:,:,3)=toeplitz(alpha(3).^(0:p-1));
%  Ct(:,:,4)=toeplitz(alpha(4).^(0:p-1));
y1=[];y2=[];
        for task=1:k-1
            X1{task}=[];y1{task}=[];
            for j=1:m
                X1{task} = [X1{task} M(:,m*(task-1)+j)+(Ct(:,:,m*(task-1)+j)^(1/2))*randn(p,ns(m*(task-1)+j))];
                y1{task}=[y1{task};j*ones(ns(m*(task-1)+j),1)];
            end
        end
        X2=[];
        for j=1:m
            X2 = [X2 M(:,m*(k-1)+j)+(Ct(:,:,j+m*(k-1))^(1/2))*randn(p,ns(m*(k-1)+j))];
            y2=[y2;j*ones(ns(m*(k-1)+j),1)];
        end
        for j=1:m
            X_test = [X_test M(:,m*(k-1)+j)+(Ct(:,:,m*(k-1)+j)^(1/2))*randn(p,nst(j))];
            y_test=[y_test;j*ones(nst(j),1)];
        end
        %X_test2=M(:,4)+randn(p,nst(1));
        S.fts=X1';T.fts=X2';
        S.labels=y1';T.labels=y2';y_test=y_test';
end
