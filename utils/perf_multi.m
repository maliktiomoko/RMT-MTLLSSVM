function [error_opt] = perf_multi(trnXs,trnXt, gamma, lambda,M,nt)
% Function that computes theoretically the means and the variance of the score for MTL
%Input:
%Output: theoretical error/Empirical error/alpha/b/Theoretical
%mean/Theoretical variance/Empirical mean/ Empirical variance
[~,n1]=size(trnXs);
[p,n2]=size(trnXt);
k=size(M,2)/2;
n=n1+n2;
co=k*p/n;
c=nt/sum(nt);
cb=zeros(k,1);
for i=1:k
    cb(i)=(nt(2*(i-1)+1)+nt(2*(i-1)+2))/sum(nt);
end
    P1=[];Z=[];
    P1=blkdiag(P1,eye(n1)-(1/n1)*ones(n1,1)*ones(1,n1));
    Z=blkdiag(Z,trnXs);
    P1=blkdiag(P1,eye(n2)-(1/n2)*ones(n2,1)*ones(1,n2));
    Z=blkdiag(Z,trnXt)*P1;
    Moy_gen2=zeros(size(M,1),1);
    Moy_gen1=zeros(size(M,1),1);
    for fr=1:2
        Moy_gen1=Moy_gen1+((nt(fr)/n1)*M(:,fr));
    end
    for fr=1:2
        Moy_gen2=Moy_gen2+((nt(2*(k-1)+fr)/n2)*M(:,2*(k-1)+fr));
    end
    vecM=[];
    vecM=[vecM Moy_gen1*ones(1,2)];
    vecM=[vecM Moy_gen2*ones(1,2)];
    M=M-vecM;
P = zeros(n, 2); 
P(1:n1,1)=ones(n1,1);P(n1+1:end,2)=ones(n2,1);
param=struct();
param.gamma=gamma;param.lambda=lambda;param.nt=[n1;n2];
[out_verif] =delta_func(p,param);
out=kron(out_verif,ones(2,1));
MM=zeros(2*k,2*k);
for i=1:k
    ei=zeros(k,1);ei(i)=1;
    Delta_mui=(M(:,2*(i-1)+1)-M(:,2*(i-1)+2));
    ci1=c(2*(i-1)+1);ci2=c(2*(i-1)+2);
    ci=ci1+ci2;
    tildeD(i)=ci/(co*(1+out(2*(i-1)+1)));
    for j=1:k
        ej=zeros(k,1);ej(j)=1;
        cj1=c(2*(j-1)+1);cj2=c(2*(j-1)+2);
        cj=cj1+cj2;
        Delta_muj=(M(:,2*(j-1)+1)-M(:,2*(j-1)+2));
        MM=MM+kron(Delta_mui'*Delta_muj*ei*ej',sqrt(ci1*ci2/(ci^3))*[sqrt(ci2);-sqrt(ci1)]*[sqrt(cj2) -sqrt(cj1)]*sqrt(cj1*cj2/(cj^3)));
        
    end
end
Agotique=(eye(k)+diag(tildeD)^(-1/2)*(diag(gamma)+lambda*ones(k,1)*ones(1,k))^(-1)*diag(tildeD)^(-1/2))^(-1);
MQ0M=kron(Agotique,ones(2,1)*ones(1,2)).*MM;
%ver=(trnY(:)-P*b);ver2=(trnY(:));
%pos=1;ytilde0=zeros(2*k,1);ytilde0(1)=ver(pos);
% pos2=1;ytilde=zeros(2*k,1);ytilde(1)=ver2(pos2);
% for i=1:2*k-1
%     pos=pos+nt(i);
%     ytilde0(i+1)=ver(pos);
%     pos2=pos2+nt(i);
%     ytilde(i+1)=ver2(pos2);
% end
tildeD=[n1;n2]./((k*p)*(1+out_verif));
Agotique1=inv(eye(k)+(diag(tildeD)^(-1/2)/(diag(gamma)+lambda*ones(k,1)*ones(1,k)))*diag(tildeD)^(-1/2));
cb=[n1;n2]/n;
DI=diag(co./(k*cb));
KAPPA=(1/k)*((Agotique1.*Agotique1)^(1/2)/(eye(k)-(Agotique1.*Agotique1)^(1/2)*DI*(Agotique1.*Agotique1)^(1/2)))*(Agotique1.*Agotique1)^(1/2);
cbar=[c(1)+c(2);c(1)+c(2);c(3)+c(4);c(3)+c(4)];
for i=1:k
    for j=1:2
        ez=zeros(k,1);ez(i)=1;cb=[cbar(1);cbar(3)];
        Vc=(1/co)*kron(Agotique*(diag((co./cb).*KAPPA(:,i)+ez))*Agotique,ones(2,1)*ones(1,2)).*MM;
    end
end
%deltabar=co*c.*out./(cbar);
 %Matm=(e1-e2)'*(eye(2*k)-diag(deltabar)^(-1/2)*Gamma*diag(deltabar)^(1/2));
 %Mat=(1./(tildeD(i)))*diag(deltabar)^(1/2)*Gamma*(diag(kron(KAPPA(:,i),ones(2,1))./cbar)+Vc)*Gamma*diag(deltabar)^(1/2);
% ytildeopt=diag(deltabar)^(1/2)*Gamma^(-1)*inv(diag(kron(KAPPA(:,i),ones(2,1))./cbar)+Vc)*(Gamma^(-1)-eye(2*k))*diag(deltabar)^(-1/2)*(e1-e2);
% G=diag(deltabar)^(1/2)*(MQ0M*(diag(kron(KAPPA(:,i),ones(2,1))./cbar)+Vc)*MQ0M)*diag(deltabar)^(1/2);
% error_opt=erfc((1/(2*sqrt(2)*sqrt(tildeD(end))))*sqrt((e1-e2)'*G*(e1-e2)));
%   Ze=[(1-nt(1)/n1) -nt(2)/n1 0 0;-nt(1)/n1 (1-nt(2)/n1) 0 0;0 0 (1-nt(3)/n2) -nt(4)/n2;0 0 (-nt(3)/n2) (1-nt(4)/n2)]; 
% obj=@(vt) -((Matm*Ze*vt)^2)./(8*vt'*Mat*vt);
%error_opt=obj(ytilde0);
deltabar=c./(co.*(1+out));
e1=zeros(2*k,1);e2=zeros(2*k,1);e1(3)=1;e2(4)=1;
G=diag(deltabar)^(-1/2)*(MQ0M*inv(diag(kron(KAPPA(:,i),ones(2,1))./cbar)+Vc)*MQ0M)*diag(deltabar)^(-1/2);
error_opt=-(((tildeD(2))))*((e1-e2)'*G*(e1-e2));
%error_opt=-(Matm/Mat)*Matm';
end