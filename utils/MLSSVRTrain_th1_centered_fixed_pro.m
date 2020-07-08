function [error_opt,error_th,error_emp,alpha, b,score_mean,variance,score_emp,var_emp,y_opt] = MLSSVRTrain_th1_centered_fixed_pro(trnXs,trnXt, trnY, gamma1,gamma2, lambda,M,Ct,J,tstX1,tstX2,tstX3,tstX4,yt1,yt2,yt3,yt4,nt,centered,veta,covariance)
% Function that computes theoretically the means and the variance of the score for MTL
%Input:
%Output: theoretical error/Empirical error/alpha/b/Theoretical
%mean/Theoretical variance/Empirical mean/ Empirical variance
switch covariance
    case 'general'
        gamma=[gamma1;gamma2];
[~,n1]=size(trnXs);
[p,n2]=size(trnXt);
k=size(M,2)/2;
n=n1+n2;
co=k*p/n;
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
    tes1=tstX1-Moy_gen1;
    %+(nt(5)/n2)*M(:,5)+(nt(6)/n2)*M(:,6));
    tstX1=[tes1;zeros(size(tes1))];
    tes2=tstX2-Moy_gen1;
    %+(nt(5)/n2)*M(:,5)+(nt(6)/n2)*M(:,6));
    tstX2=[tes2;zeros(size(tes2))];
    tes3=tstX3-Moy_gen2;
    %+(nt(5)/n2)*M(:,5)+(nt(6)/n2)*M(:,6));
    tstX3=[zeros(size(tes3).*[k-1 1]);tes3];
    tes4=tstX4-Moy_gen2;
    %+(nt(5)/n2)*M(:,5)+(nt(6)/n2)*M(:,6));
    tstX4=[zeros(size(tes4).*[k-1 1]);tes4];
    vecM=[];
    vecM=[vecM Moy_gen1*ones(1,2)];
    vecM=[vecM Moy_gen2*ones(1,2)];
    M=M-vecM;
elseif strcmp(centered,'all')
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
    tes1=tstX1(1:p,:)-Moy_gen1;
    %+(nt(5)/n2)*M(:,5)+(nt(6)/n2)*M(:,6));
    tstX1=[tes1;zeros(size(tes1))];
    tes2=tstX2(1:p,:)-Moy_gen1;
    %+(nt(5)/n2)*M(:,5)+(nt(6)/n2)*M(:,6));
    tstX2=[tes2;zeros(size(tes2))];
    tes3=tstX3((k-1)*p+1:end,:)-Moy_gen2;
    %+(nt(5)/n2)*M(:,5)+(nt(6)/n2)*M(:,6));
    tstX3=[zeros(size(tes3).*[k-1 1]);tes3];
    tes4=tstX4((k-1)*p+1:end,:)-Moy_gen2;
    %+(nt(5)/n2)*M(:,5)+(nt(6)/n2)*M(:,6));
    tstX4=[zeros(size(tes4).*[k-1 1]);tes4];
    vecM=[];
    vecM=[vecM Moy_gen1*ones(1,2)];
    vecM=[vecM Moy_gen2*ones(1,2)];
    M=M-vecM;
end

P = zeros(n, 2); 
P(1:n1,1)=ones(n1,1);P(n1+1:end,2)=ones(n2,1);
%H=(2/lambda)*(Z'*Z)+Z'*(R*R')*Z+(1/gamma)*eye(n);
A=kron(diag(gamma)+lambda*ones(k,1)*ones(1,k),eye(p));
H=Z'*A*Z+eye(n);
%A=eye(2*p);
% Dtilde=inv((A^(1/2)*(Z*Z')*A^(1/2))+eye(2*p));
%A=(2/lambda)*eye(2*p);
Mb=[];
for i=1:k
    for j=1:2
        a=zeros(k,1);a(i)=1;
        Mb=[Mb kron(a,M(:,2*(i-1)+j))];
    end
end
M1=M(:,1:2);
M2=M(:,3:4);
C=zeros(k*p,k*p,2*k);
c=zeros(2*k,1);
for i=1:k
    for j=1:2
        d=zeros(k,1);d(i)=1;d=d*d';
        C(:,:,2*(i-1)+j)=A^(1/2)*(kron(d,Ct(:,:,2*(i-1)+j)+M(:,2*(i-1)+j)*M(:,2*(i-1)+j)'))*A^(1/2);
        c(2*(i-1)+j)=nt(2*(i-1)+j)/sum(nt);
    end
end
eta = H \ P; 
nu = H \ trnY(:); 

S = P'*eta;
b = (S\eta')*trnY(:);
alpha = nu - eta*b;
tstN1 = size(tstX1, 2);
tstN2 = size(tstX2, 2);
tstN3 = size(tstX3, 2);
tstN4 = size(tstX4, 2);
score1=(1/sqrt(k*p))*(tstX1'*A*Z*alpha)+ones(tstN1,1)*b(1);
score2=(1/sqrt(k*p))*(tstX2'*A*Z*alpha)+ones(tstN2,1)*b(1);
score3=(1/sqrt(k*p))*(tstX3'*A*Z*alpha)+ones(tstN3,1)*b(2);
score4=(1/sqrt(k*p))*(tstX4'*A*Z*alpha)+ones(tstN4,1)*b(2);
% score11=(1/sqrt(2*p))*(tstX1'*A*Z*P1*alpha1)+ones(tstN,1)*b1(1);
% score12=(1/sqrt(2*p))*(tstX1'*A*Z*P2*alpha2)+ones(tstN,1)*b2(1);
% 
% score2_t=(1/(sqrt(2*p)))*tstX1'*A*Z*(H\(trnY(:)-P*b))+ones(tstN,1)*b(1);
% score1_t=(1/(sqrt(2*p)))*tstX1'*A^(1/2)*Dtilde*A^(1/2)*Z*(trnY(:)-P*b)+ones(tstN,1)*b(1);


%score2=((2/lambda)*tstX2'*Z*alpha+tstX2'*(R*R')*Z*alpha)+ones(tstN2,1)*b(1);
%score3=((2/lambda)*tstX3'*Z*alpha+tstX3'*(R*R')*Z*alpha)+ones(tstN3,1)*b(2);
%score4=((2/lambda)*tstX4'*Z*alpha+tstX4'*(R*R')*Z*alpha)+ones(tstN4,1)*b(2);
score_emp(1)=mean(score1);score_emp(2)=mean(score2);score_emp(3)=mean(score3);score_emp(4)=mean(score4);
var_emp(1)=var(score1);var_emp(2)=var(score2);var_emp(3)=var(score3);var_emp(4)=var(score4);
%var_test=(1/(k*p))*(trnY(:)-P*b)'*Z'*A^(1/2)*Ab(:,:,4)*A^(1/2)*Z*(trnY(:)-P*b)
% M110=[M1;zeros(p,2)];M220=[zeros(p,2);M2];
% if strcmp(centered,'no')
%     tstX_mean=[M110 M220];
% elseif strcmp(centered,'task')
%     tstX_mean=[M110 M220]-[c(1)*M110(:,1)+c(2)*M110(:,2) c(1)*M110(:,1)+c(2)*M110(:,2) c(3)*M220(:,1)+c(4)*M220(:,2) c(3)*M220(:,1)+c(4)*M220(:,2)];
% elseif strcmp(centered,'all')
%     tstX_mean=[M110 M220]-[c(1)*M110(:,1)+c(2)*M110(:,2)+c(3)*M220(:,1)+c(4)*M220(:,2) c(1)*M110(:,1)+c(2)*M110(:,2)+c(3)*M220(:,1)+c(4)*M220(:,2) c(1)*M110(:,1)+c(2)*M110(:,2)+c(3)*M220(:,1)+c(4)*M220(:,2) c(1)*M110(:,1)+c(2)*M110(:,2)+c(3)*M220(:,1)+c(4)*M220(:,2)];
% end
m=2;
M110=[];
for tas=1:k-1
    fi=zeros(k,1);fi(tas)=1;
    M110=[M110 kron(fi,M(:,m*(tas-1)+1:m*tas))];
end
M220=[zeros((k-1)*p,m);M2];
tstX_mean=[M110 M220];
param.gamma=gamma;param.lambda=lambda;
[out] = delta_F2(p,param,C,c,co,2,'general','synthetic');
%Mdelta = bsxfun(@rdivide, Mb, (1+out)');
Mdelta1 = bsxfun(@rdivide, Mb, (1+out)');
invQtilde=zeros(k*p,k*p);
for i=1:k
    for j=1:2
        invQtilde=invQtilde+(c(2*(i-1)+j)/co)*squeeze(C(:,:,2*(i-1)+j))/((1+out(2*(i-1)+j)));
    end
end
Qtildez=inv(invQtilde+eye(k*p));
%Qtildez1=Qtildez-(Qtildez*A^(1/2)*Mdelta1*J'*P/(k*p))*((eye(k)+P'*P-(1/gamma)*(P'/H)*P)\(P'*J*Mdelta1'*A^(1/2)*Qtildez));
% eq_mat=[];
% for i=1:k
%     for j=1:2
%         eq_mat=[eq_mat;(1/(k*p))*trace(A^(1/2)*Qtildez*A^(1/2)*C(:,:,2*(i-1)+j))*ones(nt(2*(i-1)+j),1)./out(2*(i-1)+j)];
%     end
% end
%Dtilde1=inv((A^(1/2)*(Z*P1*Z')*A^(1/2))+(1/gamma)*eye(2*p));
% Di=diag([1/n1;1/n2]);
%P2=[ones(n1,1)/sqrt(n1) zeros(n1,1);zeros(n2,1) ones(n2,1)/sqrt(n2)];
%test11=tstX1(:,1)'*A*(Z/H)*(trnY(:)-P*b);
%test12=tstX1(:,1)'*A*(Z/H)*(trnY(:)-P*b)-(1/gamma)*tstX1(:,1)'*A*(Z/H)*P2*(inv(eye(k)-P2'*P2+(1/gamma)*(P2'/H)*P2)*(P2'/H))*(trnY(:)-P*b);
%Dtilde_comp=inv(inv(Dtilde)-A^(1/2)*Z*P*Di*P'*Z'*A^(1/2));
%Dtilde_comp=Dtilde+(Dtilde*A^(1/2)*Z*P*Di^(1/2))*((eye(k)-Di^(1/2)*P'*Z'*A^(1/2)*Dtilde*A^(1/2)*Z*P*Di^(1/2))\(Di^(1/2)*P'*Z'*A^(1/2)*Dtilde));
%Dtilde_comp=Dtilde+(Dtilde*A^(1/2)*Z*P*Di^(1/2))*((eye(k)-Di^(1/2)*P'*P*Di^(1/2)+(1/gamma)*(Di^(1/2)*P'/H)*P*Di^(1/2))\(Di^(1/2)*P'*Z'*A^(1/2)*Dtilde));
%Eq_mat=P'*Z'*A^(1/2)*Dtilde*A^(1/2)*Z;
%Eq_mat=(1/(k*p))*P'*J*Mdelta1'*A^(1/2)*Qtildez*A^(1/2)*Mdelta1*J'+P'*diag(eq_mat);
%Equival=A^(1/2)*Qtildez*A^(1/2)*Mdelta1*J'+A^(1/2)*(Qtildez*A^(1/2)*Mdelta1*J'*P)*((eye(k)-P'*P+(1/gamma)*(P'/H)*P)\(Eq_mat));
%Equival2=A^(1/2)*Dtilde1*A^(1/2)*Z*P1;
%Equival=A^(1/2)*Qtildez*A^(1/2)*Mdelta1*J'*P1/sqrt(k*p)+(1/sqrt(k*p))*A^(1/2)*(Qtildez*A^(1/2)*Mdelta1*J'*P*Di^(1/2))*((eye(k)-Di^(1/2)*P'*P*Di^(1/2)+(1/gamma)*(Di^(1/2)*P'/H)*P*Di^(1/2))\(Di^(1/2)*Eq_mat))*P1;
% score11=tstX1'*Equival*(trnY(:)-P*b)/sqrt(k*p);
% score12=tstX1'*Equival2*(trnY(:)-P*b)/(sqrt(k*p));
%ones(1,k*p)*Qtildez1*ones(k*p,1)/(k*p)
%score_test=(1/(sqrt(k*p)))*tstX1'*A^(1/2)*Dtilde*A^(1/2)*Z*(trnY(:)-P*b)
%teste=(1/(k*p))*Mgota'*Qtildez*Mgota*(J'*(trnY(:)-P*b)./((1+out).*sqrt(are)))./(sqrt(are))+[b(1);b(1);b(2);b(2)];
score_mean=(1/(k*p))*tstX_mean'*A^(1/2)*Qtildez*A^(1/2)*Mdelta1*J'*(trnY(:)-P*b)+[b(1);b(1);b(2);b(2)];
ea=zeros(k,1);ea(1)=1;
%M_got_t=kron(((2/lambda*eye(k)+ones(k,1)*ones(1,k))^(1/2))*ea,M(:,1));
Mgot=[];are=zeros(2*k,1);aren=zeros(k,1);
for i=1:k
    for j=1:2
        ei=zeros(k,1);ei(i)=1;
        Mgot=[Mgot kron(((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2))*ei,M(:,2*(i-1)+j))];
        are(2*(i-1)+j)=(nt(2*(i-1)+j)/(k*p*(1+out(2*(i-1)+j))));
    end
    aren(i)=are(2*(i-1)+1)+are(2*(i-1)+2);
end
% invQ2=zeros(k*p,k*p);
% for i=1:k
%     d=zeros(k,1);d(i)=1;%d=d*d';
%     %invQ2=invQ2+aren(i)*((kron(((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2)*d*((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2),eye(p))));
%     for j=1:2
%         %invQtilde=invQtilde+c(2*(i-1)+j)*squeeze(C(:,:,2*(i-1)+j))/(co*(1+out(2*(i-1)+j)));
%         invQ2=invQ2+sqrt(are(2*(i-1)+j))*kron(((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2)*d,M(:,2*(i-1)+j))*sqrt(are(2*(i-1)+j))*kron(d'*((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2),M(:,2*(i-1)+j)');
%     end
% end
% invQ2=invQ2+((kron(((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2)*diag(aren)*((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2),eye(p))));
%Qtest2=inv(invQ2+(1/gamma)*eye(k*p));
Mgota=bsxfun(@rdivide, Mgot, 1./sqrt(are)');
%Qtest1=inv()
%INT=inv((((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2))*diag(aren)*(((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2))+(1/gamma)*eye(k));
%Qtilde0=kron(INT,eye(p));vr=1;br=1;
invQtilde0=zeros(k*p,k*p);
for i=1:k
    e=zeros(k,1);e(i)=1;
    invQtilde0=invQtilde0+kron(((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2)*e*e'*((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2))),are(2*(i-1)+1)*Ct(:,:,2*(i-1)+1)+are(2*(i-1)+2)*Ct(:,:,2*(i-1)+2));
end
Qtilde0=inv(invQtilde0+eye(k*p));
%inter=Mgota'*Qtilde0*Mgota;ei=zeros(k,1);ei(vr)=1;ej=zeros(k,1);ej(br)=1;
%Df=inv((2/lambda)*diag((aren.^2))+(1/gamma)*diag(aren));
%Df1=inv((2/lambda)*diag((aren))+(1/gamma)*eye(k));con=1/(1+ones(1,k)*diag(aren)*Df1*ones(k,1));
%rty=Df-Df1*ones(k,1)*ones(1,k)*Df1*con;
%inter_el=kron(ei'*(diag(1./aren)-(1/gamma)*Df)*ej,M(:,br)'*M(:,vr))*(sqrt(are(br)*are(vr)))+...
%    kron(ei'*((1/gamma)*Df1*ones(k,1)*ones(1,k)*Df1*con)*ej,M(:,br)'*M(:,vr))*(sqrt(are(br)*are(vr)));
Qtest=Qtilde0-Qtilde0*Mgota*inv(eye(2*k)+Mgota'*Qtilde0*Mgota)*Mgota'*Qtilde0;
ver=(trnY(:)-P*b);
pos=1;ytilde=zeros(2*k,1);ytilde(1)=ver(pos);
for i=1:2*k-1
    pos=pos+nt(i);
    ytilde(i+1)=ver(pos);
end
v=nt'.*ytilde./(sqrt(are).*(1+out));
vp=v;
%M_got_t=kron((((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2))*ea,M(:,1));
%score_test=(1/(sqrt(k*p)))*tstX3'*A^(1/2)*Dtilde*A^(1/2)*Z*(trnY(:)-P*b)+b(2);
%score3=(1/sqrt(2*p))*((2/lambda)*tstX3'*Z*alpha+tstX3'*(R*R')*Z*alpha)+ones(tstN,1)*b(2);
S1=zeros(k*p,k*p,2*k);d1=zeros(2*k,1);
for i=1:k
    for j=1:2
        d1(2*(i-1)+j)=nt(2*(i-1)+j)/(k*p*((1+out(2*(i-1)+j))^2));
        e=zeros(k,1);e(i)=1;e=e*e';
        S1(:,:,2*(i-1)+j)=kron(e,Ct(:,:,2*(i-1)+j));
    end
end
%var1=(1/(2*p))*((trnY(:)-P*b)'/H)*Z'*A*S1(:,:,1)*A*Z*(H\(trnY(:)-P*b));
%var11=(1/(2*p))*((trnY(:)-P*b)'/H)*P1*Z'*A*S1(:,:,1)*A*Z*P1*(H\(trnY(:)-P*b));
%var12=(1/(2*p))*((trnY(:)-P*b)'/H)*P2*Z'*A*S1(:,:,1)*A*Z*P2*(H\(trnY(:)-P*b));
%variance_centred=(1/(k*p))*(trnY(:)-P*b)'*P1*Z'*A^(1/2)*Dtilde1*A^(1/2)*S1(:,:,1)*A^(1/2)*Dtilde1*A^(1/2)*Z*P1*(trnY(:)-P*b);
%variance_centred1=(1/(k*p))*(trnY(:)-P*b)'*P1*Z'*A^(1/2)*Dtilde*A^(1/2)*S1(:,:,1)*A^(1/2)*Dtilde*A^(1/2)*Z*P1*(trnY(:)-P*b)+...
%    (1/(k*p))*(trnY(:)-P*b)'*P1*Z'*A^(1/2)*Dtilde*A^(1/2)*S1(:,:,1)*A^(1/2)*(Dtilde*A^(1/2)*Z*P*Di^(1/2))*((eye(k)-Di^(1/2)*P'*P*Di^(1/2)+(1/gamma)*(Di^(1/2)*P'/H)*P*Di^(1/2))\(Di^(1/2)*P'*Z'*A^(1/2)*Dtilde))*A^(1/2)*Z*P1*(trnY(:)-P*b)+...
%    (1/(k*p))*(trnY(:)-P*b)'*P1*Z'*A^(1/2)*(Dtilde*A^(1/2)*Z*P*Di^(1/2))*((eye(k)-Di^(1/2)*P'*P*Di^(1/2)+(1/gamma)*(Di^(1/2)*P'/H)*P*Di^(1/2))\(Di^(1/2)*P'*Z'*A^(1/2)*Dtilde))*A^(1/2)*S1(:,:,1)*A^(1/2)*Dtilde*A^(1/2)*Z*P1*(trnY(:)-P*b)+...
%    (1/(k*p))*(trnY(:)-P*b)'*P1*Z'*A^(1/2)*(Dtilde*A^(1/2)*Z*P*Di^(1/2))*((eye(k)-Di^(1/2)*P'*P*Di^(1/2)+(1/gamma)*(Di^(1/2)*P'/H)*P*Di^(1/2))\(Di^(1/2)*P'*Z'*A^(1/2)*Dtilde))*A^(1/2)*S1(:,:,1)*A^(1/2)*(Dtilde*A^(1/2)*Z*P*Di^(1/2))*((eye(k)-Di^(1/2)*P'*P*Di^(1/2)+(1/gamma)*(Di^(1/2)*P'/H)*P*Di^(1/2))\(Di^(1/2)*P'*Z'*A^(1/2)*Dtilde))*A^(1/2)*Z*P1*(trnY(:)-P*b);

D=diag(d1);
TT=zeros(2*k,2*k);
for i=1:k
    for j=1:2
        for l=1:k
            for m=1:2
                TT(2*(i-1)+j,2*(l-1)+m)=(1/(k*p))*trace(C(:,:,2*(i-1)+j)*Qtildez*C(:,:,2*(l-1)+m)*Qtildez);
            end
        end
    end
end
T=TT/(eye(2*k)-D*TT);
for i=1:k
    for j=1:2
        TS=zeros(k*p,k*p);
        for l=1:k
            for m=1:2
                TS=TS+d1(2*(l-1)+m)*T(2*(i-1)+j,2*(l-1)+m)*Qtildez*squeeze(C(:,:,2*(l-1)+m))*Qtildez;
            end
        end
        Tto(:,:,2*(i-1)+j)=TS;
    end
end
B=zeros(k*p,k*p,2*k);
TB=zeros(k*p,k*p,2*k);
for i=1:k
    for j=1:2
        B(:,:,2*(i-1)+j)=Qtildez*C(:,:,2*(i-1)+j)*Qtildez+Tto(:,:,2*(i-1)+j);
        %B(:,:,2*(i-1)+j)=Dtilde*C(:,:,2*(i-1)+j)*Dtilde;
    end
end
for i=1:k
    for j=1:2
        TI=zeros(k*p,k*p);
        for l=1:k
            for m=1:2
                TI=TI+d1(2*(l-1)+m)*trace(A^(1/2)*squeeze(S1(:,:,2*(i-1)+j))*A^(1/2)*Qtildez*squeeze(C(:,:,2*(l-1)+m))*Qtildez)*squeeze(B(:,:,2*(l-1)+m))/(k*p);
            end
        end
        TB(:,:,2*(i-1)+j)=TI;
    end
end
Ab=zeros(k*p,k*p,2*k);Cg=[];
for i=1:k
    for j=1:2
        %f=zeros(2*k,1);f(2*(i-1)+j)=1;
        Ab(:,:,2*(i-1)+j)=Qtildez*A^(1/2)*S1(:,:,2*(i-1)+j)*A^(1/2)*Qtildez+TB(:,:,2*(i-1)+j);
        %jt=J(:,2*(i-1)+j);
        %Cg=[Cg;jt'*trace(Ab(:,:,2*(i-1)+j)*A^(-1/2)*C(:,:,2*(i-1)+j))/(k*p*(1+(1/co)*out(2*(i-1)+j)))];
        %vecC=[vecC;ones(nt(2*(i-1)+j),1)*trace(C(:,:,2*(i-1)+j)*Ab(:,:,2*(i-1)+j))/(1+out(2*(i-1)+j))];
        %Ep=Ep+(1/(k*p))*trace(C(:,:,2*(i-1)+j)*Ab(:,:,2*(i-1)+j))*f*f';
    end
end
% for i=1:k
%     for j=1:2
%         Ab(:,:,2*(i-1)+j)=Dtilde*A^(1/2)*S1(:,:,2*(i-1)+j)*A^(1/2)*Dtilde;
%     end
% end
for i=1:k
    for j=1:2
        VCg=[];VEp=[];
        for l=1:k
            for m=1:2
                f=zeros(2*k,1);f(2*(l-1)+m)=1;
                jt=J(:,2*(l-1)+m);
                VCg=[VCg;jt'*trace(Ab(:,:,2*(i-1)+j)*C(:,:,2*(l-1)+m))/(k*p*(1+out(2*(l-1)+m)))];
                %vecC=[vecC;ones(nt(2*(i-1)+j),1)*trace(C(:,:,2*(i-1)+j)*Ab(:,:,2*(i-1)+j))/(1+out(2*(i-1)+j))];
                VEp=[VEp;(trace(C(:,:,2*(l-1)+m)*Ab(:,:,2*(i-1)+j))/((1+out(2*(l-1)+m)).^2))*ones(nt(2*(l-1)+m),1)];
            end
        end
        Cg(:,:,2*(i-1)+j)=VCg;
        Ep(:,:,2*(i-1)+j)=diag(VEp);
    end
end
%Cg=diag(vecC)/(p);
variance=zeros(2*k,1);
for i=1:k
    for j=1:2
        variance(2*(i-1)+j)=(1/(k*p)^2)*(trnY(:)-P*b)'*(J*Mdelta1'*A^(1/2)*Ab(:,:,2*(i-1)+j)*A^(1/2)*Mdelta1*J'+Ep(:,:,2*(i-1)+j))*(trnY(:)-P*b)-...
            (2/(k*p)^2)*(trnY(:)-P*b)'*J*Mdelta1'*A^(1/2)*Qtildez*A^(1/2)*Mdelta1*Cg(:,:,2*(i-1)+j)*(trnY(:)-P*b);
    end
end
%test1=(1/(k*p)^2)*(trnY(:)-P*b)'*(J*Mdelta1'*A^(1/2)*Ab(:,:,2*(i-1)+j)*A^(1/2)*Mdelta1*J')*(trnY(:)-P*b);
%test1=(1/(k*p)^2)*v'*(Mgota'*Ab(:,:,2*(i-1)+j)*Mgota)*v;
Gamma=inv(eye(2*k)+Mgota'*Qtilde0*Mgota);
Mat_var=zeros(k*p,k*p,k*2);
y_op=zeros(2*k,2*k);
for i=1:k
    for j=1:2
        N2=zeros(k*p,k*p);
        var3=0;
        for l=1:k
            for m=1:2
                kappa(2*(l-1)+m)=(d1(2*(l-1)+m)/(k*p))*trace(A^(1/2)*squeeze(S1(:,:,2*(i-1)+j))*A^(1/2)*Qtildez*squeeze(C(:,:,2*(l-1)+m))*Qtildez);
                N3=zeros(k*p,k*p);
                for n=1:k
                    for o=1:2
                        zeta(2*(l-1)+m,2*(n-1)+o)=d1(2*(n-1)+o)*T(2*(l-1)+m,2*(n-1)+o);
                        N3=N3+zeta(2*(l-1)+m,2*(n-1)+o)*squeeze(C(:,:,2*(n-1)+o));
                    end
                end
                N2=N2+kappa(2*(l-1)+m)*(C(:,:,2*(l-1)+m)+N3);
                etilde(2*(l-1)+m,2*(i-1)+j)=trace(C(:,:,2*(l-1)+m)*Ab(:,:,2*(i-1)+j))/(k*p*(1+out(2*(l-1)+m)));
                %vec3(2*(l-1)+m)=are(2*(l-1)+m).*trace(C(:,:,2*(l-1)+m)*Ab(:,:,2*(i-1)+j))./nt(2*(l-1)+m);
            end
        end
        Mat_var(:,:,2*(i-1)+j)=A^(1/2)*S1(:,:,2*(i-1)+j)*A^(1/2)+N2;
        Mat_var2(:,:,2*(i-1)+j)=(eye(2*k)-Gamma)*diag(etilde(:,2*(i-1)+j));
        zti(:,2*(i-1)+j)=(nt'.*ytilde.*etilde(:,2*(i-1)+j)./((1+out).*sqrt(are)));
        Mat(:,:,2*(i-1)+j)=(Gamma*(Mgota'*Qtilde0*Mat_var(:,:,2*(i-1)+j)*Qtilde0*Mgota)*Gamma-(Mat_var2(:,:,2*(i-1)+j)+Mat_var2(:,:,2*(i-1)+j)')+diag(etilde(:,2*(i-1)+j)));
        %Mat=toeplitz(0.1.^(0:2*k-1));
        e1=zeros(2*k,1);e2=zeros(2*k,1);e1(3)=1./sqrt(are(3));e2(4)=1./sqrt(are(4));
        Matm=(e1-e2)'*(eye(2*k)-Gamma);
        %opt_max=Mat\(Matm');
        variance_test1(2*(i-1)+j)=(1/(k*p)^2)*v'*Mat(:,:,2*(i-1)+j)*v;
        %y_op(:,2*(i-1)+j)=opt_max.*sqrt(are).*(1+out)./(nt');
        %variance_test11(2*(i-1)+j)=(1/(k*p)^2)*opt_max'*Mat*opt_max;
    end
end
threshold=sqrt(2)*erfcinv(2*veta);
threshold2=-sqrt(variance(3))*threshold+score_mean(3);
func=@(x) -(-threshold*sqrt(x'*Mat(:,:,3)*x)+x'*Matm')./sqrt(x'*Mat(:,:,4)*x);
x0=[1;-1;1;-1];
[opt_max,err]=fminunc(func,x0);opt_max=opt_max./norm(opt_max);
Ze=[(1-nt(1)/n1) -nt(2)/n1 0 0;-nt(1)/n1 (1-nt(2)/n1) 0 0;0 0 (1-nt(3)/n2) -nt(4)/n2;0 0 (-nt(3)/n2) (1-nt(4)/n2)];
y_opt=pinv(Ze)*(opt_max.*sqrt(are).*(1+out)./(nt'));
score_mean_test=(1/(k*p))*v'*(eye(2*k)-Gamma)*diag(sqrt(1./are))+[b(1) b(1) b(2) b(2)];
error_opt=0.5*erfc(real(sqrt(abs((Matm/Mat(:,:,3))*Matm'/8))));
%inr=((score_mean_test)/(4*2*(abs(variance_test1(3)))));
in1=((score_mean(1)-score_mean(2))/(2*sqrt(2)*sqrt(abs(variance(1)))));
in2=((score_mean(1)-score_mean(2))/(2*sqrt(2)*sqrt(abs(variance(2)))));
%in3=((score_mean(3)-score_mean(4))/(2*sqrt(2)*sqrt(abs(variance(3)))));

in4=(threshold2-score_mean(4))/sqrt(abs(variance(4)));
in3=(score_mean(3)-threshold2)/sqrt(abs(variance(3)));
error_th(1)=0.5*erfc(real(in1)/sqrt(2));
error_th(2)=0.5*erfc(real(in2)/(sqrt(2)));
error_th(3)=0.5*erfc(real(in3)/(sqrt(2)));
error_th(4)=0.5*erfc(real(in4)/(sqrt(2)));
pred1=zeros(size(tstX1,2),1);pred2=zeros(size(tstX2,2),1);pred3=zeros(size(tstX3,2),1);pred4=zeros(size(tstX4,2),1);
pred1(score1>((mean(score1)+mean(score2))/2))=1;pred1(score1<((mean(score1)+mean(score2))/2))=-1;
pred2(score2>((mean(score1)+mean(score2))/2))=1;pred2(score2<((mean(score1)+mean(score2))/2))=-1;
pred3(score3>threshold2)=1;pred3(score3<threshold2)=-1;
pred4(score4>threshold2)=1;pred4(score4<threshold2)=-1;
error_emp(3)=sum(pred3~=yt3)/size(tstX3,2);
error_emp(1)=sum(pred1~=yt1)/size(tstX1,2);
error_emp(2)=sum(pred2~=yt2)/size(tstX2,2);
error_emp(4)=sum(pred4~=yt4)/size(tstX4,2);
    case 'identity'
        gamma=[gamma1;gamma2];
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
            tes1=tstX1(1:p,:)-Moy_gen1;
            tstX1=[tes1;zeros(size(tes1))];
            tes2=tstX2(1:p,:)-Moy_gen1;
            tstX2=[tes2;zeros(size(tes1))];
            tes3=tstX3((k-1)*p+1:end,:)-Moy_gen2;
            tstX3=[zeros(size(tes3).*[k-1 1]);tes3];
            tes4=tstX4((k-1)*p+1:end,:)-Moy_gen2;
            tstX4=[zeros(size(tes4).*[k-1 1]);tes4];
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
            tstX2=tstX2-[trnXtotc;zeros(size(trnXtotc))]*ones(n,1)/n;
            tstX3=tstX3-[zeros(size(trnXtotc));trnXtotc]*ones(n,1)/n;
            tstX4=tstX4-[zeros(size(trnXtotc));trnXtotc]*ones(n,1)/n;
            M=M-[c(1)*M(:,1)+c(2)*M(:,2)+c(3)*M(:,3)+c(4)*M(:,4) c(1)*M(:,1)+c(2)*M(:,2)+c(3)*M(:,3)+c(4)*M(:,4) c(1)*M(:,1)+c(2)*M(:,2)+c(3)*M(:,3)+c(4)*M(:,4) c(1)*M(:,1)+c(2)*M(:,2)+c(3)*M(:,3)+c(4)*M(:,4)];
        end

        P = zeros(n, 2); 
        P(1:n1,1)=ones(n1,1);P(n1+1:end,2)=ones(n2,1);
        A=kron((diag(gamma)+lambda*ones(k,1)*ones(1,k)),eye(p));
        H=(Z'*A*Z+eye(n));
        Mb=[];
        for i=1:k
            for j=1:2
                a=zeros(k,1);a(i)=1;
                Mb=[Mb kron(a,M(:,2*(i-1)+j))];
            end
        end
        M1=M(:,1:2);
        M2=M(:,3:4);
        C=zeros(k*p,k*p,2*k);
        for i=1:k
            for j=1:2
                d=zeros(k,1);d(i)=1;d=d*d';
                C(:,:,2*(i-1)+j)=A^(1/2)*(kron(d,Ct(:,:,2*(i-1)+j)+M(:,2*(i-1)+j)*M(:,2*(i-1)+j)'))*A^(1/2);
            end
        end
        eta = H \ P; 
        nu = H \ trnY(:); 

        S = P'*eta;
        b = (S\eta')*trnY(:);
        alpha = nu - eta*b;
        tstN1 = size(tstX1, 2);
        tstN2 = size(tstX2, 2);
        tstN3 = size(tstX3, 2);
        tstN4 = size(tstX4, 2);
        score1=(1/sqrt(k*p))*(tstX1'*A*Z*alpha)+ones(tstN1,1)*b(1);
%         score11=((1/sqrt(k*p))*tstX_mean(:,1)'*A*Z*alpha2)+b(1);
        score2=(1/sqrt(k*p))*(tstX2'*A*Z*alpha)+ones(tstN2,1)*b(1);
        score3=(1/sqrt(k*p))*(tstX3'*A*Z*alpha)+ones(tstN3,1)*b(2);
        score4=(1/sqrt(k*p))*(tstX4'*A*Z*alpha)+ones(tstN4,1)*b(2);
        score_emp(1)=mean(score1);score_emp(2)=mean(score2);score_emp(3)=mean(score3);score_emp(4)=mean(score4);
        var_emp(1)=var(score1);var_emp(2)=var(score2);var_emp(3)=var(score3);var_emp(4)=var(score4);
        M110=[M1;zeros(p,2)];M220=[zeros(p,2);M2];
        tstX_mean=[M110 M220];
        param=struct();
        param.gamma=gamma;param.lambda=lambda;param.nt=[n1;n2];
        [out_verif] = delta_F(p,param,C,c,co,2,'identity','synthetic');
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
        vz=(sqrt(c)'.*ytilde0./(sqrt((1+out))));
        for i=1:k
            for j=1:2
                ez=zeros(k,1);ez(i)=1;cb=[cbar(1);cbar(3)];
                Vc=(1/co)*kron(Agotique*(diag((co./cb).*KAPPA(:,i)+ez))*Agotique,ones(2,1)*ones(1,2)).*MM;
                variance(2*(i-1)+j)=(1./(tildeD(i)))*((vz'*Gamma*(diag(kron(KAPPA(:,i),ones(2,1))./cbar)+Vc)*Gamma*vz));
            end
        end
        deltabar=co*c'.*out./(cbar);
        score_mean=ytilde-diag(deltabar)^(-1/2)*Gamma*diag(deltabar)^(1/2)*ytilde0;
        e1=zeros(2*k,1);e2=zeros(2*k,1);e1(3)=1;e2(4)=1;
        Matm=(e1-e2)'*(eye(2*k)-diag(deltabar)^(-1/2)*Gamma*diag(deltabar)^(1/2));
        Mat=(1./(tildeD(i)))*Gamma*(diag(kron(KAPPA(:,i),ones(2,1))./cbar)+Vc)*Gamma;
        obj=@(vt) -((Matm*vt)^2)./(8*vt'*Mat*vt);
         x0=[1;-1;1;-1];
         [obj1,error_opt]=fminunc(obj,x0);
         Ze=[(1-nt(1)/n1) -nt(2)/n1 0 0;-nt(1)/n1 (1-nt(2)/n1) 0 0;0 0 (1-nt(3)/n2) -nt(4)/n2;0 0 (-nt(3)/n2) (1-nt(4)/n2)];
         y_op=pinv(Ze)*(obj1./(sqrt(c)'.*ytilde0./(sqrt((1+out)))));
        in1=((score_mean(1)-score_mean(2))/(2*sqrt(2)*sqrt(abs(variance(1)))));
        in2=((score_mean(1)-score_mean(2))/(2*sqrt(2)*sqrt(abs(variance(2)))));
        in3=((score_mean(3)-score_mean(4))/(2*sqrt(2)*sqrt(abs(variance(3)))));
        in4=((score_mean(3)-score_mean(4))/(2*sqrt(2)*sqrt(abs(variance(4)))));
        error_th(1)=0.5*erfc(real(in1));
        error_th(2)=0.5*erfc(real(in2));
        error_th(3)=0.5*erfc(real(in3));
        error_th(4)=0.5*erfc(real(in4));
        pred1=zeros(size(tstX1,2),1);pred2=zeros(size(tstX2,2),1);pred3=zeros(size(tstX3,2),1);pred4=zeros(size(tstX4,2),1);
        pred1(score1>((mean(score1)+mean(score2))/2))=1;pred1(score1<((mean(score1)+mean(score2))/2))=-1;
        pred2(score2>((mean(score1)+mean(score2))/2))=1;pred2(score2<((mean(score1)+mean(score2))/2))=-1;
        pred3(score3>((mean(score3)+mean(score4))/2))=1;pred3(score3<((mean(score3)+mean(score4))/2))=-1;
        pred4(score4>((mean(score3)+mean(score4))/2))=1;pred4(score4<((mean(score3)+mean(score4))/2))=-1;
        error_emp(3)=sum(pred3~=yt3)/size(tstX3,2);
        error_emp(1)=sum(pred1~=yt1)/size(tstX1,2);
        error_emp(2)=sum(pred2~=yt2)/size(tstX2,2);
        error_emp(4)=sum(pred4~=yt4)/size(tstX4,2);
end
