function [error,score_mean,variance_th] = MLSSVRTrain_th1_centered_other_class_one_vs_one(trnXs,trnXt,trnY, gamma, lambda,M,Ct,tstX1,nt,centered,k,n_test,Mt,mt,J,Ctot,ntot)
% Function that computes theoretically the means and the variance of the score for MTL
%Input:
%Output: theoretical error/Empirical error/alpha/b/Theoretical
%mean/Theoretical variance/Empirical mean/ Empirical variance
n1=[];
for task=1:k-1
    [~,n11{task}]=size(trnXs{task});
    n1=[n1;n11{task}];
end
[p,n2]=size(trnXt);
nd=[n1;n2];
m=size(Ct,3)/k;
% for i=1:length(alpha)
%     Ct(:,:,i)=alpha(i)*eye(p);
% end
trnY(trnY==2)=-1;
%Ct(:,:,1)=eye(p);Ct(:,:,2)=eye(p);Ct(:,:,3)=eye(p);Ct(:,:,4)=eye(p);
n=sum(nd);
co=k*p/n;
c=nt/sum(nt);
% cb=zeros(k,1);
% for i=1:k
%     cb(i)=(nt(2*(i-1)+1)+nt(2*(i-1)+2))/sum(nt);
% end
if strcmp(centered,'no')
    Z=[trnXs zeros(size(trnXt));zeros(size(trnXs)) trnXt];
elseif strcmp(centered,'task')
    P1=[];Z=[];
    for task=1:k-1
        P1=blkdiag(P1,eye(n11{task})-(1/n11{task})*ones(n11{task},1)*ones(1,n11{task}));
        Z=blkdiag(Z,trnXs{task});
    end
    P1=blkdiag(P1,eye(n2)-(1/n2)*ones(n2,1)*ones(1,n2));
    Z=blkdiag(Z,trnXt)*P1;
    Moy_gen2=zeros(size(M,1),1);
    vecM=[];
    for task=1:k-1
        Moy_gen1{task}=zeros(size(M,1),1);
        for fr=1:m
            Moy_gen1{task}=Moy_gen1{task}+((nt(m*(task-1)+fr)/n11{task})*M(:,m*(task-1)+fr));
        end
        vecM=[vecM Moy_gen1{task}*ones(1,m)];
    end
    for fr=1:m
        Moy_gen2=Moy_gen2+((nt(m*(k-1)+fr)/n2)*M(:,m*(k-1)+fr));
    end
    tes1=tstX1-Moy_gen2;
    tstX1=[zeros(size(tes1).*[k-1 1]);tes1];
    vecM=[vecM Moy_gen2*ones(1,m)];
    M=M-vecM;
    vecMt=[Moy_gen1{task}*ones(1,mt) Moy_gen2*ones(1,mt)];
    tstX_mean=[zeros(p,2*mt);Mt-vecMt];
    Mtot=Mt-vecMt;
elseif strcmp(centered,'all')
    trnXtot=[trnXs trnXt];
    P1=(eye(n)-(1/n)*ones(n,1)*ones(1,n));
    trnXtotc=trnXtot*P1;
    Z=[trnXtotc(:,1:n1) zeros(p,n2);zeros(p,n1) trnXtotc(:,n1+1:end)];
    tstX1=tstX1-[trnXtotc;zeros(size(trnXtotc))]*ones(n,1)/n;
    M=M-[c(1)*M(:,1)+c(2)*M(:,2)+c(3)*M(:,3)+c(4)*M(:,4) c(1)*M(:,1)+c(2)*M(:,2)+c(3)*M(:,3)+c(4)*M(:,4) c(1)*M(:,1)+c(2)*M(:,2)+c(3)*M(:,3)+c(4)*M(:,4) c(1)*M(:,1)+c(2)*M(:,2)+c(3)*M(:,3)+c(4)*M(:,4)];
end
% Z=scale_data(Z,'1',p);
%X2=scale_data(X2,'1');
P = zeros(n, k); 
P(1:n11{1},1)=ones(n11{1},1);
if k>2
    for task=1:k-2
        P(1+sum([n11{1:task}]):sum([n11{1:task+1}]),task+1)=ones(n11{task+1},1);
    end
end
P(sum([n11{1:k-1}])+1:end,k)=ones(n2,1);
%A=kron((diag(gamma)+lambda*ones(k,1)*ones(1,k)),eye(p));
A=kron((diag(gamma)+sqrt(lambda)*sqrt(lambda)'),eye(p));
lambda=lambda./norm(A);gamma=gamma./norm(A);
%A=kron((diag(gamma)+lambda*ones(k,1)*ones(1,k)),eye(p));
A=kron((diag(gamma)+sqrt(lambda)*sqrt(lambda)'),eye(p));
H=(Z'*A*Z+eye(n));
  Dtilde=(A^(1/2)*Z*Z'*A^(1/2)+eye(k*p))^(-1);
% Xc1=[];Xm11=[];Xm21=[];M111=[];M112=[];
% c1(1)=1;c1(2)=2;M11=[];M22=[];c2(1)=1;c2(2)=2;
% Ct=zeros(size(trnXs,1),size(trnXs,1),2*k);
% for task=1:k-1
%     for i=1:m
%       X11{i,task}=trnXs(:,ysr==c1(i))';
% %         c12{i}=X11{i,task}(1:floor(nt(2*(task-1)+i)/3),:);
% %         Xc1=[Xc1 c12{i}];
% %         Xm11=[Xm11 X11{i,task}(floor(nt(2*(task-1)+i)/3)+1:2*floor(nt(2*(task-1)+i)/3),:)];
% %         Xm21=[Xm21 X11{i,task}(2*floor(nt(2*(task-1)+i)/3)+1:end,:)];
% %         X1=[X1 X11{i,task}'];
%         M11=[M11 mean(X11{i,task})'];
%         M111=[M111 mean(X11{i,task}(floor(nt(2*(task-1)+i)/3)+1:2*floor(nt(2*(task-1)+i)/3),:))'];
%         M112=[M112 mean(X11{i,task}(2*floor(nt(2*(task-1)+i)/3)+1:end,:))'];
% %         Ct(:,:,m*(task-1)+i)=(k*p)*((c12{i}-mean(c12{i},1))'*(c12{i}-mean(c12{i},1)))/floor(size(c12{i},1));
% %          Ct(:,:,m*(task-1)+i)=(k*p)*((X11{i,task}-mean(X11{i,task},1))'*(X11{i,task}-mean(X11{i,task},1)))/nt(m*(task-1)+i);
% 
%     end
% end
%Xc2=[];Xm12=[];Xm22=[];M211=[];M212=[];
%for i=1:m
    %X22{i}=trnXt(:,ytar==c2(i))';
%     c2{i}=X22{i}(1:floor(nt(2*(k-1)+i)/3),:);
%     ns(i+m*(k-1))=floor(size(X22{i},1)/2);
    %M22=[M22 mean(X22{i})'];
    %Xm222{i}=X22{i}(2*floor(nt(2*(k-1)+i)/3)+1:end,:);
    %Xm122{i}=X22{i}(floor(nt(2*(k-1)+i)/3)+1:2*floor(nt(2*(k-1)+i)/3),:);
    %M211=[M211 mean(Xm122{i})'];
    %M212=[M212 mean(Xm222{i})'];
%     Xc2=[Xc2 c2{i}];
%     Xm12=[Xm12 X22{i}(floor(nt(2*(k-1)+i)/3)+1:2*floor(nt(2*(k-1)+i)/3),:)];
%     Xm22=[Xm22 X22{i}(2*floor(nt(2*(k-1)+i)/3)+1:end,:)];
%     Ct(:,:,i+m*(k-1))=(k*p)*((c2{i}-mean(c2{i},1))'*(c2{i}-mean(c2{i},1)))/floor(size(c2{i},1));
%     Ct(:,:,i+m*(k-1))=(k*p)*((X22{i}-mean(X22{i},1))'*(X22{i}-mean(X22{i},1)))/nt(i+m*(k-1));
 
%end
%     M=[M11 M22]*(sqrt(k*p));
%  Mc1=[M111 M211];
% Mc2=[M112 M212];
%A=eye(2*p);
%A=(2/lambda)*eye(2*p);
Mb=[];
for i=1:k
    for j=1:2
        a=zeros(k,1);a(i)=1;
        Mb=[Mb kron(a,M(:,2*(i-1)+j))];
    end
end
M1=M(:,1:m*(k-1));
 M2=M(:,m*(k-1)+1:end);
%  Ct(:,:,1)=eye(p); Ct(:,:,2)=eye(p); Ct(:,:,3)=eye(p); Ct(:,:,4)=eye(p);
C=zeros(k*p,k*p,m*k);
for i=1:k-1
    for j=1:m
        d=zeros(k,1);d(i)=1;d=d*d';
         C(:,:,m*(i-1)+j)=A^(1/2)*(kron(d,Ct(:,:,m*(i-1)+j)+M(:,m*(i-1)+j)*M(:,m*(i-1)+j)'))*A^(1/2);
%         C(:,:,m*(i-1)+j)=(k*p)*A^(1/2)*(kron(d,X11{j,i}'*X11{j,i}/nt(m*(i-1)+j)))*A^(1/2);
    end
end
    for j=1:m
        d=zeros(k,1);d(k)=1;d=d*d';
         C(:,:,m*(k-1)+j)=A^(1/2)*(kron(d,Ct(:,:,m*(k-1)+j)+M(:,m*(k-1)+j)*M(:,m*(k-1)+j)'))*A^(1/2);
%         C(:,:,m*(k-1)+j)=(k*p)*A^(1/2)*(kron(d,X22{j}'*X22{j}/nt(m*(k-1)+j)))*A^(1/2);
    end
eta = H \ P; 
nu = H \ trnY(:); 
 
S = P'*eta;
b = (S\eta')*trnY(:);
alpha2 = nu - eta*b;
 tstN1 = size(tstX1, 2);
% tstN2 = size(tstX2, 2);
% tstN3 = size(tstX3, 2);
% tstN4 = size(tstX4, 2);
score1=(1/sqrt(k*p))*(tstX1'*A*Z*alpha2)+ones(tstN1,1)*b(k);
% score2=(1/sqrt(2*p))*(tstX2'*A*Z*alpha2)+ones(tstN2,1)*b(1);
% score3=(1/sqrt(2*p))*(tstX3'*A*Z*alpha2)+ones(tstN3,1)*b(2);
% score4=(1/sqrt(2*p))*(tstX4'*A*Z*alpha2)+ones(tstN4,1)*b(2);
% score11=(1/sqrt(2*p))*(tstX1'*A*Z*P1*alpha1)+ones(tstN,1)*b1(1);
% score12=(1/sqrt(2*p))*(tstX1'*A*Z*P2*alpha2)+ones(tstN,1)*b2(1);
% 
% score2_t=(1/(sqrt(2*p)))*tstX1'*A*Z*(H\(trnY(:)-P*b))+ones(tstN,1)*b(1);
% score1_t=(1/(sqrt(2*p)))*tstX1'*A^(1/2)*Dtilde*A^(1/2)*Z*(trnY(:)-P*b)+ones(tstN,1)*b(1);
%score2=(1/sqrt(2*p))*((2/lambda)*tstX2'*Z*alpha2+tstX2'*(R*R')*Z*alpha2)+ones(tstN2,1)*b(1);
%score3=(1/sqrt(2*p))*((2/lambda)*tstX3'*Z*alpha2+tstX3'*(R*R')*Z*alpha2)+ones(tstN3,1)*b(2);
%score4=(1/sqrt(2*p))*((2/lambda)*tstX4'*Z*alpha2+tstX4'*(R*R')*Z*alpha2)+ones(tstN4,1)*b(2);
score_emp(1)=mean(score1(1:n_test(1)));
for i=1:m-1
score_emp(i+1)=mean(score1(sum(n_test(1:i))+1:sum(n_test(1:i+1))));
end
%score_emp(2)=mean(score2);score_emp(3)=mean(score3);score_emp(4)=mean(score4);
var_emp(1)=var(score1(1:n_test(1)));
for i=1:m-1
var_emp(i+1)=var(score1(sum(n_test(1:i))+1:sum(n_test(1:i+1))));
end
%var_emp(2)=var(score2);var_emp(3)=var(score3);var_emp(4)=var(score4);
%var_test=(1/(k*p))*(trnY(:)-P*b)'*Z'*A^(1/2)*Ab(:,:,4)*A^(1/2)*Z*(trnY(:)-P*b)
M110=zeros(k*p,m*(k-1));
for task=1:k-1
    rg_n{task}=p*(task-1)+1:p*task;
    for j=1:m
        M110(rg_n{task},m*(task-1)+j)=M1(:,m*(task-1)+j);
    end
%     M110(rg_n{task},m*(task-1)+2)=M1(:,m*(task-1)+2);
end
M220=[zeros(p*(k-1),m);M2];
M0=[M110 M220];
param=struct();
param.gamma=gamma;param.lambda=lambda;param.nt=[n1;n2];
[out] = delta_F(p,param,C,c,co,m,'synthetic');
out_verif=[];
for task=1:k-1
    out_verif=[out_verif;out(m*(task-1)+1)];
end
 Mdelta1 = bsxfun(@rdivide, Mb, (1+out)');
invQtilde=zeros(k*p,k*p);
for i=1:k
    for j=1:m
         invQtilde=invQtilde+(c(m*(i-1)+j)/co)*squeeze(C(:,:,m*(i-1)+j))/((1+out(m*(i-1)+j)));
%         invQtilde=invQtilde+(c(2*(i-1)+j)/co)*squeeze(C(:,:,2*(i-1)+j));
    end
end
Qtildez=inv(invQtilde+eye(k*p));
score_mean=(1/(k*p))*tstX_mean'*A^(1/2)*Qtildez*A^(1/2)*Mdelta1*J'*(trnY(:)-P*b)+[b(1)*ones(mt,1);b(2)*ones(mt,1)];
for task=1:k-1 
    X1{task}=trnXs{task};
end
 X1{k}=trnXt;
are=c./(co*(1+out));
Mgota=A^(1/2)*M0*diag(sqrt(are));
 invQtilde0=zeros(k*p,k*p);
 for g=1:k
     e=zeros(k,1);e(g)=1;
%      invQtilde0=invQtilde0+kron(((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2)*e*e'*((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2))),are(2*(g-1)+1)*Ct(:,:,2*(g-1)+1)+are(2*(g-1)+2)*Ct(:,:,2*(g-1)+2));
        Mat=zeros(p,p);
        for j=1:m
            Mat=Mat+are(m*(g-1)+j)*Ct(:,:,m*(g-1)+j);
        end
%      invQtilde0=invQtilde0+kron(((diag(gamma)+sqrt(lambda)*sqrt(lambda)')^(1/2)*e*e'*((diag(gamma)+sqrt(lambda)*sqrt(lambda)')^(1/2))),are(2*(g-1)+1)*Ct(:,:,2*(g-1)+1)+are(2*(g-1)+2)*Ct(:,:,2*(g-1)+2));
     invQtilde0=invQtilde0+kron(((diag(gamma)+sqrt(lambda)*sqrt(lambda)')^(1/2)*e*e'*((diag(gamma)+sqrt(lambda)*sqrt(lambda)')^(1/2))),Mat);
     %invQtilde0=invQtilde0+kron(((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2)*e*e'*((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2))),Ct(:,:,2*(g-1)+1)+Ct(:,:,2*(g-1)+2));
 end
 Qtilde0=inv(invQtilde0+eye(k*p));
MQ0M=Mgota'*Qtilde0*Mgota;
%MQ0Mc1=Mgotac1'*Qtilde0*Mgotac2;
%  MQ0M(1:2,1:2)=MQ0Mc1(1:2,1:2);
%  MQ0M(3:4,3:4)=MQ0Mc1(3:4,3:4);
Gamma=inv(eye(m*k)+MQ0M);
% Gamma2=inv(eye(2*k)+MQ0M1);
ver=(trnY(:)-P*b);ver2=(trnY(:));
pos=1;ytilde0=zeros(m*k,1);ytilde0(1)=ver(pos);
pos2=1;ytilde=zeros(m*k,1);ytilde(1)=ver2(pos2);
for i=1:m*k-1
    pos=pos+nt(i);
    ytilde0(i+1)=ver(pos);
    pos2=pos2+nt(i);
    ytilde(i+1)=ver2(pos2);
end
% cbar=[c(1)+c(2);c(1)+c(2);c(3)+c(4);c(3)+c(4)];
% deltabar=co*c.*out./(cbar);
 deltabar=c./(co.*(1+out));
 score_th=ytilde-diag(deltabar)^(-1/2)*Gamma*diag(deltabar)^(1/2)*ytilde0;
 Co=zeros(k*p,k*p,m*k);
for i=1:k-1
    for j=1:mt
        d=zeros(k,1);d(i)=1;d=d*d';
         Co(:,:,mt*(i-1)+j)=A^(1/2)*(kron(d,Ctot(:,:,mt*(i-1)+j)+Mtot(:,mt*(i-1)+j)*Mtot(:,mt*(i-1)+j)'))*A^(1/2);
%         C(:,:,m*(i-1)+j)=(k*p)*A^(1/2)*(kron(d,X11{j,i}'*X11{j,i}/nt(m*(i-1)+j)))*A^(1/2);
    end
end
    for j=1:mt
        d=zeros(k,1);d(k)=1;d=d*d';
         Co(:,:,mt*(k-1)+j)=A^(1/2)*(kron(d,Ctot(:,:,mt*(k-1)+j)+Mtot(:,mt*(k-1)+j)*Mtot(:,mt*(k-1)+j)'))*A^(1/2);
%         C(:,:,m*(k-1)+j)=(k*p)*A^(1/2)*(kron(d,X22{j}'*X22{j}/nt(m*(k-1)+j)))*A^(1/2);
    end
%     ctot=ntot/sum(ntot);
%     cot=k*p/sum(ntot);
% [outo] = delta_F(p,param,Co,ctot,cot,mt,'synthetic');
S1=zeros(k*p,k*p,m*k);
for i=1:k
    for j=1:m
        e=zeros(k,1);e(i)=1;e=e*e';
        S1(:,:,m*(i-1)+j)=kron(e,Ct(:,:,m*(i-1)+j));
    end
end
S11=zeros(k*p,k*p,mt*k);
for i=1:k
    for j=1:mt
        e=zeros(k,1);e(i)=1;e=e*e';
        S11(:,:,mt*(i-1)+j)=kron(e,Ctot(:,:,mt*(i-1)+j));
    end
end
for i=1:k
    for j=1:m
        d1(m*(i-1)+j)=nt(m*(i-1)+j)/(k*p*((1+out(m*(i-1)+j))^2));
    end
end
D=diag(d1);
TT=zeros(mt*k,m*k);
for i=1:k
    for j=1:mt
        for l=1:k
            for mc=1:m
                TT(mt*(i-1)+j,m*(l-1)+mc)=(1/(k*p))*trace(Co(:,:,mt*(i-1)+j)*Qtildez*C(:,:,m*(l-1)+mc)*Qtildez);
            end
        end
    end
end
Ck=zeros(m*k,m*k);
for i=1:k
    for j=1:m
        for l=1:k
            for mc=1:m
                Ck(m*(i-1)+j,m*(l-1)+mc)=(1/(k*p))*trace(C(:,:,m*(i-1)+j)*Qtildez*C(:,:,m*(l-1)+mc)*Qtildez);
            end
        end
    end
end
T=TT/(eye(m*k)-D*Ck);kappa=zeros(mt*k,m*k);V=zeros(k*p,k*p,mt*k);
for i=1:k
    for j=1:mt
%         TS=zeros(k*p,k*p);
        TS2=zeros(k*p,k*p);
        for l=1:k
            for mc=1:m
%                 TS=TS+d1(2*(l-1)+m)*T(2*(i-1)+j,2*(l-1)+m)*Qtildez*squeeze(C(:,:,2*(l-1)+m))*Qtildez;
                kappa(mt*(i-1)+j,2*(l-1)+mc)=d1(2*(l-1)+mc)*T(mt*(i-1)+j,2*(l-1)+mc);
                TS2=TS2+kappa(mt*(i-1)+j,2*(l-1)+mc)*A^(1/2)*S1(:,:,2*(l-1)+mc)*A^(1/2);
                
            end
        end
%         Tto(:,:,2*(i-1)+j)=TS;
        V(:,:,mt*(i-1)+j)=A^(1/2)*S11(:,:,mt*(i-1)+j)*A^(1/2)+TS2;
    end
end
Mat=zeros(m*k,m*k,mt*k);
for i=1:k
    for j=1:mt
%         MGMc1=Mgotac1'*Qtilde0*Mgotac2;
        MGM=Mgota'*Qtilde0*V(:,:,mt*(i-1)+j)*Qtilde0*Mgota;
%         MGM(1:2,1:2)=MGMc1(1:2,1:2);MGM(3:4,3:4)=MGMc1(3:4,3:4);
        Mat(:,:,mt*(i-1)+j)=diag(sqrt(are))*Gamma*(MGM+diag(kappa(2*(i-1)+j,:)./are'))*Gamma*diag(sqrt(are));
        variance_th(mt*(i-1)+j)=ytilde0'*Mat(:,:,mt*(i-1)+j)*ytilde0;
    end
end
moy=(score_th(3)+score_th(4))/2;
for i=1:mt
    error(i)=0.5*erfc(sqrt((moy-score_mean(mt+i))^2./(2*variance_th(mt+i))));
end
%   figure
%   for ht=1:mt
%    x{ht} = score_mean(mt*(k-1)+ht)+sqrt(variance_th(mt*(k-1)+ht))*[-3:.1:3];
%    y{ht} = normpdf(x{ht},score_mean(mt*(k-1)+ht),sqrt(variance_th(mt*(k-1)+ht)));
%    hold all
%    plot(x{ht},y{ht}./sum(y{ht}),'LineWidth',3);
%   end
%     hold on
%   histogram(real(score1(1:n_test(1),1)),80,'Normalization','probability')
%   for i=1:mt-1
%     histogram(real(score1(sum(n_test(1:i))+1:sum(n_test(1:i+1)),1)),80,'Normalization','probability')
%   end
  
end
