function [score,error_th,alpha2, b,score_th,variance_th,score_emp,var_emp,y_opt] = MLSSVRTrain_th1_centered(trnXs,trnXt, trnY, gamma, lambda,M,tstX1,nt,centered)
% Function that computes theoretically the means and the variance of the score for MTL
%Input:
%Output: theoretical error/Empirical error/alpha/b/Theoretical
%mean/Theoretical variance/Empirical mean/ Empirical variance
[~,n1]=size(trnXs);
[p,n2]=size(trnXt);
%Ct(:,:,1)=eye(p);Ct(:,:,2)=eye(p);Ct(:,:,3)=eye(p);Ct(:,:,4)=eye(p);
k=size(M,2)/2;
n=n1+n2;
co=k*p/n;
Ct(:,:,1)=eye(p);Ct(:,:,2)=eye(p);Ct(:,:,3)=eye(p);Ct(:,:,4)=eye(p);
c=nt/sum(nt);
cb=zeros(k,1);
for i=1:k
    cb(i)=(nt(2*(i-1)+1)+nt(2*(i-1)+2))/sum(nt);
end
if strcmp(centered,'no')
    Z=[trnXs zeros(size(trnXt));zeros(size(trnXs)) trnXt];
elseif strcmp(centered,'task')
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
    tes1=tstX1-Moy_gen2;
    tstX1=[zeros(size(tes1));tes1];
%      tes2=tstX2-Moy_gen1;
%      tstX2=[tes2;zeros(size(tes1))];
%      tes3=tstX3-Moy_gen2;
%      tstX3=[zeros(size(tes3).*[k-1 1]);tes3];
%      tes4=tstX4-Moy_gen2;
%      tstX4=[zeros(size(tes4).*[k-1 1]);tes4];
    vecM=[];
    vecM=[vecM Moy_gen1*ones(1,2)];
    vecM=[vecM Moy_gen2*ones(1,2)];
    M=M-vecM;
elseif strcmp(centered,'all')
    trnXtot=[trnXs trnXt];
    P1=(eye(n)-(1/n)*ones(n,1)*ones(1,n));
    trnXtotc=trnXtot*P1;
    Z=[trnXtotc(:,1:n1) zeros(p,n2);zeros(p,n1) trnXtotc(:,n1+1:end)];
    tstX1=tstX1-[trnXtotc;zeros(size(trnXtotc))]*ones(n,1)/n;
%     tstX2=tstX2-[trnXtotc;zeros(size(trnXtotc))]*ones(n,1)/n;
%     tstX3=tstX3-[zeros(size(trnXtotc));trnXtotc]*ones(n,1)/n;
%     tstX4=tstX4-[zeros(size(trnXtotc));trnXtotc]*ones(n,1)/n;
    M=M-[c(1)*M(:,1)+c(2)*M(:,2)+c(3)*M(:,3)+c(4)*M(:,4) c(1)*M(:,1)+c(2)*M(:,2)+c(3)*M(:,3)+c(4)*M(:,4) c(1)*M(:,1)+c(2)*M(:,2)+c(3)*M(:,3)+c(4)*M(:,4) c(1)*M(:,1)+c(2)*M(:,2)+c(3)*M(:,3)+c(4)*M(:,4)];
end

P = zeros(n, 2); 
P(1:n1,1)=ones(n1,1);P(n1+1:end,2)=ones(n2,1);
A=kron((diag(gamma)+lambda*ones(k,1)*ones(1,k)),eye(p));
H=(Z'*A*Z+eye(n));
% Mb=[];
% for i=1:k
%     for j=1:2
%         a=zeros(k,1);a(i)=1;
%         Mb=[Mb kron(a,M(:,2*(i-1)+j))];
%     end
% end
% C=zeros(k*p,k*p,2*k);
% for i=1:k
%     for j=1:2
%         d=zeros(k,1);d(i)=1;d=d*d';
%         C(:,:,2*(i-1)+j)=A^(1/2)*(kron(d,Ct(:,:,2*(i-1)+j)+M(:,2*(i-1)+j)*M(:,2*(i-1)+j)'))*A^(1/2);
%     end
% end
% M1=M(:,1:2);
% M2=M(:,3:4);
eta = H \ P; 
nu = H \ trnY(:); 

S = P'*eta;
b = (S\eta')*trnY(:);
alpha2 = nu - eta*b;
tstN1 = size(tstX1, 2);
%  tstN2 = size(tstX2, 2);
%  tstN3 = size(tstX3, 2);
%  tstN4 = size(tstX4, 2);
score=(1/sqrt(2*p))*(tstX1'*A*Z*alpha2)+ones(tstN1,1)*b(2);
%  score2=(1/sqrt(2*p))*(tstX2'*A*Z*alpha2)+ones(tstN2,1)*b(2);
%  score3=(1/sqrt(2*p))*(tstX3'*A*Z*alpha2)+ones(tstN3,1)*b(2);
%  score4=(1/sqrt(2*p))*(tstX4'*A*Z*alpha2)+ones(tstN4,1)*b(2);
score_emp(1)=mean(score);
%  score11=mean(score(1:1000));score12=mean(score(1001:end));
%  var11=var(score(1:1000));var12=var(score(1001:end));
% score_emp(2)=mean(score2);score_emp(3)=mean(score3);score_emp(4)=mean(score4);
var_emp(1)=var(score);
% var_emp(2)=var(score2);var_emp(3)=var(score3);var_emp(4)=var(score4);
%   M110=[M1;zeros(p,2)];M220=[zeros(p,2);M2];
%   tstX_mean=[M110 M220];
param=struct();
param.gamma=gamma;param.lambda=lambda;param.nt=[n1;n2];
%[out2] = delta_F(p,param,C,c,co,2,'synthetic');
[out_verif] =delta_func(p,param);
out=kron(out_verif,ones(2,1));
% [out,Qtildez] = delta_F(p,param,C,c,co,2,'synthetic');
%  Dtilde=inv((A^(1/2)*(Z*Z')*A^(1/2))+eye(k*p));
%  Mdelta1 = bsxfun(@rdivide, Mb, (1+out)');
%  invQtilde=zeros(k*p,k*p);
%  for i=1:k
%      for j=1:2
%          invQtilde=invQtilde+(c(2*(i-1)+j)/co)*squeeze(C(:,:,2*(i-1)+j))/((1+out(2*(i-1)+j)));
%      end
%  end
%  Qtildez=inv(invQtilde+eye(k*p));
 J=zeros(n,2*k);
for i=1:2*k
    J(sum(nt(1:i-1))+1:sum(nt(1:i)),i)=ones(nt(i),1);
end
score=(1/sqrt(2*p))*(tstX1'*A*Z*alpha2)+ones(tstN1,1)*b(2);
% score11=(1/sqrt(2*p))*(tstX1'*A*Z*(H\(trnY(:)-P*b)))+ones(tstN1,1)*b(2);
% a=Dtilde*A^(1/2)*Z(:,1);
% Zi=Z;Zi(:,1)=[];Zi2=Z;Zi2(:,end)=[];
% Dtildei=inv((A^(1/2)*(Zi*Zi')*A^(1/2))+eye(k*p));
% Dtildei2=inv((A^(1/2)*(Zi2*Zi2')*A^(1/2))+eye(k*p));
% b1=Dtildei*A^(1/2)*Z(:,1)./(1+Z(:,1)'*A^(1/2)*Dtildei*A^(1/2)*Z(:,1))
% det=Z(:,51)'*A^(1/2)*Dtildei*A^(1/2)*Z(:,51);
% det4=Z(:,end)'*A^(1/2)*Dtildei2*A^(1/2)*Z(:,end)
% det2=Z(:,1)'*A^(1/2)*Dtildei*A^(1/2)*Z(:,1)
% det23=trace(Dtildei2*A^(1/2)*Z(:,end)*Z(:,end)'*A^(1/2))
% det3=trace(Dtildei*C(:,:,1))/(k*p)
% %out(1)=det2;out(1)=det2;out(3)=det4;out(4)=det4;
%    score_test11=(1/sqrt(k*p))*tstX1(:,1:1000)'*A^(1/2)*Dtilde*A^(1/2)*Z*(trnY(:)-P*b)+b(2);
%   score_test1=(1/(k*p))*tstX1(:,1:1000)'*A^(1/2)*Qtildez*A^(1/2)*Mdelta1*J'*(trnY(:)-P*b)+b(2);
%   score_mean=(1/(k*p))*tstX_mean'*A^(1/2)*Qtildez*A^(1/2)*Mdelta1*J'*(trnY(:)-P*b)+[b(1);b(1);b(2);b(2)];
%  score_mean=(1/(k*p))*tstX_mean'*A^(1/2)*Qtildez*A^(1/2)*Mdelta1*J'*(trnY(:)-P*b)+[b(1);b(1);b(2);b(2)];
 
%  %%%%%%%%%%%%%%%%%%% A SUPPRIMER %%%%%%%%%%%%%%%%%%%%%%
%  Mgot=[];
%  for i=1:k
%     for j=1:2
%         ei=zeros(k,1);ei(i)=1;
%         Mgot=[Mgot kron(((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2))*ei,M(:,2*(i-1)+j))];
%         are(2*(i-1)+j)=(nt(2*(i-1)+j)/(k*p*(1+out(2*(i-1)+j))));
%     end
%     aren(i)=are(2*(i-1)+1)+are(2*(i-1)+2);
%  end
% Mgota=bsxfun(@rdivide, Mgot, 1./sqrt(are));
%  ver=(trnY(:)-P*b);ver2=(trnY(:));
% pos=1;ytilde0=zeros(2*k,1);ytilde0(1)=ver(pos);
% pos2=1;ytilde=zeros(2*k,1);ytilde(1)=ver2(pos2);
% for i=1:2*k-1
%     pos=pos+nt(i);
%     ytilde0(i+1)=ver(pos);
%     pos2=pos2+nt(i);
%     ytilde(i+1)=ver2(pos2);
% end
% v=nt.*ytilde./(sqrt(are').*(1+out));
% 
%  Qtilde0=kron(inv((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2)*diag(aren)*((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2))+eye(k)),eye(p));
% Qtest=Qtilde0-Qtilde0*Mgota*inv(eye(2*k)+Mgota'*Qtilde0*Mgota)*Mgota'*Qtilde0;
% Gamma1=inv(eye(2*k)+Mgota'*Qtilde0*Mgota);
% score_mean1=(1/(k*p))*tstX_mean'*A^(1/2)*Qtilde0*A^(1/2)*Mdelta1*J'*(trnY(:)-P*b)-...
%     (1/(k*p))*tstX_mean'*A^(1/2)*Qtilde0*Mgota*inv(eye(2*k)+Mgota'*Qtilde0*Mgota)*Mgota'*Qtilde0*A^(1/2)*Mdelta1*J'*(trnY(:)-P*b)+[b(1);b(1);b(2);b(2)]
% score_mean_test=(1/(k*p))*v'*(eye(2*k)-Gamma1)*diag(sqrt(1./are'))+[b(1) b(1) b(2) b(2)];
% va=c.*ytilde./(co*(1+out));
% v/(k*p)
% vze=(sqrt(c).*ytilde./(sqrt(co*(1+out))));
% deltart=sqrt(c)./(sqrt(co*(1+out)));
% deltasr=c./(co.*(1+out))
% score43=ytilde'*diag(deltart)*(eye(2*k)-Gamma1)*diag(1./deltart)
% ytilde-diag(deltasr)^(-1/2)*Gamma1*diag(deltasr)^(1/2)*ytilde
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 
MM=zeros(2*k,2*k);tildeD=zeros(k,1);
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
ver=(trnY(:)-P*b);ver2=(trnY(:));
pos=1;ytilde0=zeros(2*k,1);ytilde0(1)=ver(pos);
pos2=1;ytilde=zeros(2*k,1);ytilde(1)=ver2(pos2);
for i=1:2*k-1
    pos=pos+nt(i);
    ytilde0(i+1)=ver(pos);
    pos2=pos2+nt(i);
    ytilde(i+1)=ver2(pos2);
end
tildeD=[n1;n2]./((k*p)*(1+out_verif));
Agotique1=inv(eye(k)+(diag(tildeD)^(-1/2)/(diag(gamma)+lambda*ones(k,1)*ones(1,k)))*diag(tildeD)^(-1/2));
cb=[n1;n2]/n;
DI=diag(co./(k*cb));
KAPPA=(1/k)*((Agotique1.*Agotique1)^(1/2)/(eye(k)-(Agotique1.*Agotique1)^(1/2)*DI*(Agotique1.*Agotique1)^(1/2)))*(Agotique1.*Agotique1)^(1/2);
cbar=[c(1)+c(2);c(1)+c(2);c(3)+c(4);c(3)+c(4)];
Gamma=inv(eye(2*k)+MQ0M);
vz=(sqrt(c).*ytilde0./(sqrt((1+out))));
for i=1:k
    for j=1:2
        ez=zeros(k,1);ez(i)=1;cb=[cbar(1);cbar(3)];
        Vc=(1/co)*kron(Agotique*(diag((co./cb).*KAPPA(:,i)+ez))*Agotique,ones(2,1)*ones(1,2)).*MM;
        variance_th(2*(i-1)+j)=(1./(tildeD(i)))*((vz'*Gamma*(diag(kron(KAPPA(:,i),ones(2,1))./cbar)+Vc)*Gamma*vz));
    end
end
%deltabar=co*c.*out./(cbar);
deltabar=c./(co.*(1+out));
%deltart=1./(sqrt(c)./(sqrt(co*(1+out))));
%score43=ytilde'*diag(deltart)*(eye(2*k)-Gamma1)*diag(1./deltart)
%score_th=ytilde-diag(deltart)^(-1/2)*Gamma*diag(deltart)^(1/2)*ytilde0;
score_th=ytilde-diag(deltabar)^(-1/2)*Gamma*diag(deltabar)^(1/2)*ytilde0;
e1=zeros(2*k,1);e2=zeros(2*k,1);e1(3)=1;e2(4)=1;
Matm=(e1-e2)'*(eye(2*k)-diag(deltabar)^(-1/2)*Gamma*diag(deltabar)^(1/2));
Mat2=e1'*(eye(2*k)-diag(deltabar)^(-1/2)*Gamma*diag(deltabar)^(1/2));
Mat=(1./(tildeD(i)))*diag(deltabar)^(1/2)*Gamma*(diag(kron(KAPPA(:,i),ones(2,1))./cbar)+Vc)*Gamma*diag(deltabar)^(1/2);
ytildeopt=diag(deltabar)^(-1/2)*Gamma^(-1)*inv(diag(kron(KAPPA(:,i),ones(2,1))./cbar)+Vc)*(Gamma^(-1)-eye(2*k))*(e1-e2);
G=diag(deltabar)^(-1/2)*(MQ0M*inv(diag(kron(KAPPA(:,i),ones(2,1))./cbar)+Vc)*MQ0M)*diag(deltabar)^(-1/2);
error_opt1=((tildeD(end)))*((e1-e2)'*G*(e1-e2))/8;
  Ze=[(1-nt(1)/n1) -nt(2)/n1 0 0;-nt(1)/n1 (1-nt(2)/n1) 0 0;0 0 (1-nt(3)/n2) -nt(4)/n2;0 0 (-nt(3)/n2) (1-nt(4)/n2)]; 
obj=@(vt) -((Matm*Ze*vt)^2)./(8*vt'*Mat*vt);
  x0=[1;-1;1;-1];
  [obj1,error_opt]=fmincon(obj,x0,[],[],[Mat2*Ze],[1]);
  y_opt=Ze*(obj1);
in1=((score_th(1)-score_th(2))/(2*sqrt(2)*sqrt(abs(variance_th(1)))));
in2=((score_th(1)-score_th(2))/(2*sqrt(2)*sqrt(abs(variance_th(2)))));
in3=((score_th(3)-score_th(4))/(2*sqrt(2)*sqrt(abs(variance_th(3)))));
in4=((score_th(3)-score_th(4))/(2*sqrt(2)*sqrt(abs(variance_th(4)))));
error_th(1)=0.5*erfc(real(in1));
error_th(2)=0.5*erfc(real(in2));
error_th(3)=0.5*erfc(real(in3));
error_th(4)=0.5*erfc(real(in4));
% m=2;
% figure
% for ht=1:m
%  x{ht} = score_th(m*(k-1)+ht)+sqrt(variance_th(m*(k-1)+ht))*[-3:.1:3];
%  yx{ht} = normpdf(x{ht},score_th(m*(k-1)+ht),sqrt(variance_th(m*(k-1)+ht)));
%  hold all
%  plot(x{ht},yx{ht}./sum(yx{ht}),'LineWidth',3);
% end
%   hold on
% histogram(real(score(1:sum(nst(1)))),80,'Normalization','probability');
%   for ht=1:10-1
%   histogram(real(score(1+sum(nst(1:ht)):sum(nst(1:ht+1)))),80,'Normalization','probability');
%   end
% pred1=zeros(size(tstX1,2),1);pred2=zeros(size(tstX2,2),1);pred3=zeros(size(tstX3,2),1);pred4=zeros(size(tstX4,2),1);
% pred1(score1>((mean(score1)+mean(score2))/2))=1;pred1(score1<((mean(score1)+mean(score2))/2))=-1;
% pred2(score2>((mean(score1)+mean(score2))/2))=1;pred2(score2<((mean(score1)+mean(score2))/2))=-1;
% pred3(score3>((mean(score3)+mean(score4))/2))=1;pred3(score3<((mean(score3)+mean(score4))/2))=-1;
% pred4(score4>((mean(score3)+mean(score4))/2))=1;pred4(score4<((mean(score3)+mean(score4))/2))=-1;
% error_emp(3)=sum(pred3~=yt3)/size(tstX3,2);
% error_emp(1)=sum(pred1~=yt1)/size(tstX1,2);
% error_emp(2)=sum(pred2~=yt2)/size(tstX2,2);
% error_emp(4)=sum(pred4~=yt4)/size(tstX4,2);
end