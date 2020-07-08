function [covar] = Covariance_calculus_one_vs_one(trnXs,trnXt,nt,M,Ct,gamma,lambda,obj3,tl,k)
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
n=sum(nd);
co=k*p/n;
c=nt/sum(nt);
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
vecM=[vecM Moy_gen2*ones(1,m)];
M=M-vecM;
A=kron((diag(gamma)+sqrt(lambda)*sqrt(lambda)'),eye(p));
lambda=lambda./norm(A);gamma=gamma./norm(A);
A=kron((diag(gamma)+sqrt(lambda)*sqrt(lambda)'),eye(p));
M1=M(:,1:m*(k-1));
 M2=M(:,m*(k-1)+1:end);
C=zeros(k*p,k*p,m*k);
for i=1:k-1
    for j=1:m
        d=zeros(k,1);d(i)=1;d=d*d';
         C(:,:,m*(i-1)+j)=A^(1/2)*(kron(d,Ct(:,:,m*(i-1)+j)+M(:,m*(i-1)+j)*M(:,m*(i-1)+j)'))*A^(1/2);
    end
end
    for j=1:m
        d=zeros(k,1);d(k)=1;d=d*d';
         C(:,:,m*(k-1)+j)=A^(1/2)*(kron(d,Ct(:,:,m*(k-1)+j)+M(:,m*(k-1)+j)*M(:,m*(k-1)+j)'))*A^(1/2);
    end
M110=zeros(k*p,m*(k-1));
for task=1:k-1
    rg_n{task}=p*(task-1)+1:p*task;
    for j=1:m
        M110(rg_n{task},m*(task-1)+j)=M1(:,m*(task-1)+j);
    end
end
M220=[zeros(p*(k-1),m);M2];
M0=[M110 M220];
 
param=struct();
param.gamma=gamma;param.lambda=lambda;param.nt=[n1;n2];
[out] = delta_F(p,param,C,c,co,m,'synthetic');
invQtilde=zeros(k*p,k*p);
for i=1:k
    for j=1:m
         invQtilde=invQtilde+(c(m*(i-1)+j)/co)*squeeze(C(:,:,m*(i-1)+j))/((1+out(m*(i-1)+j)));
    end
end
Qtildez=inv(invQtilde+eye(k*p));
are=c./(co*(1+out));
Mgota=A^(1/2)*M0*diag(sqrt(are));
 invQtilde0=zeros(k*p,k*p);
 for g=1:k
     e=zeros(k,1);e(g)=1;
        Mat=zeros(p,p);
        for j=1:m
            Mat=Mat+are(m*(g-1)+j)*Ct(:,:,m*(g-1)+j);
        end
     invQtilde0=invQtilde0+kron(((diag(gamma)+sqrt(lambda)*sqrt(lambda)')^(1/2)*e*e'*((diag(gamma)+sqrt(lambda)*sqrt(lambda)')^(1/2))),Mat);
 end
 Qtilde0=inv(invQtilde0+eye(k*p));
MQ0M=Mgota'*Qtilde0*Mgota;
Gamma=inv(eye(m*k)+MQ0M);
S1=zeros(k*p,k*p,m*k);d1=zeros(m*k,1);
for i=1:k
    for j=1:m
        d1(m*(i-1)+j)=nt(m*(i-1)+j)/(k*p*((1+out(m*(i-1)+j))^2));
        e=zeros(k,1);e(i)=1;e=e*e';
        S1(:,:,m*(i-1)+j)=kron(e,Ct(:,:,m*(i-1)+j));
    end
end
D=diag(d1);
TT=zeros(m*k,m*k);
for i=1:k
    for j=1:m
        for l=1:k
            for mc=1:m
                TT(m*(i-1)+j,m*(l-1)+mc)=(1/(k*p))*trace(C(:,:,m*(i-1)+j)*Qtildez*C(:,:,m*(l-1)+mc)*Qtildez);
            end
        end
    end
end
T=TT/(eye(m*k)-D*TT);kappa=zeros(m*k,m*k);V=zeros(k*p,k*p,m*k);
Rep=zeros(k*p,k*p,m*k);
for i=1:k
    e=zeros(k,1);e(i)=1;
    rep1=(diag(gamma)+sqrt(lambda)*sqrt(lambda)')^(1/2)*e*e'*(diag(gamma)+sqrt(lambda)*sqrt(lambda)')^(1/2);
    for j=1:m
        Rep(:,:,m*(i-1)+j)=kron(rep1,Ct(:,:,m*(i-1)+j));
    end
end
for i=1:k
    for j=1:m
        TS2=zeros(k*p,k*p);
        for l=1:k
            for mc=1:m
                kappa(m*(i-1)+j,m*(l-1)+mc)=d1(m*(l-1)+mc)*T(m*(i-1)+j,m*(l-1)+mc);
                TS2=TS2+kappa(m*(i-1)+j,m*(l-1)+mc)*Rep(:,:,m*(l-1)+mc);
                
            end
        end
        V(:,:,m*(i-1)+j)=Rep(:,:,m*(i-1)+j)+TS2;
    end
end
covar=zeros(m-1,m-1);
for kc=1:m-1
        ytildevarO=obj3(:,kc);
    for j=1:m-1
        ytildevar0=obj3(:,j);
        MGM=Mgota'*Qtilde0*V(:,:,m*(k-1)+tl)*Qtilde0*Mgota;
    %         MGM(1:2,1:2)=MGMc1(1:2,1:2);MGM(3:4,3:4)=MGMc1(3:4,3:4);
        Mat3=diag(sqrt(are))*Gamma*(MGM+diag(kappa(m*(k-1)+tl,:)./are'))*Gamma*diag(sqrt(are));
            %variance_th(m*(i-1)+j)=
        covar(kc,j)=ytildevarO'*Mat3*ytildevar0;
    end
end
end
