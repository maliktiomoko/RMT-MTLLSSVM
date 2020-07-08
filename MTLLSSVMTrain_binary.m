function [score1,error_opt,error_th,error_emp,alpha2, b,score_th,variance_th,score_emp,var_emp,y_opt,pred1,obj1] = MTLLSSVMTrain_binary(trnXs,trnXt,trnY, gamma, lambda,M,Ct,tstX1,nt,centered,k,n_test)
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
m=2
%m=size(Ct,3)/k;
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
% Mb=[];
% for i=1:k
%     for j=1:2
%         a=zeros(k,1);a(i)=1;
%         Mb=[Mb kron(a,M(:,2*(i-1)+j))];
%     end
% end
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
%M110c1=[M111;zeros(p,2)];M220c1=[zeros(p,2);M211];
%M110c2=[M112;zeros(p,2)];M220c2=[zeros(p,2);M212];
M0=[M110 M220];
%M0c1=[M110c1 M220c1]*sqrt(k*p);
%M0c2=[M110c2 M220c2]*sqrt(k*p);
% if strcmp(centered,'no')
%     tstX_mean=[M110 M220];
% elseif strcmp(centered,'task')
%     tstX_mean=[M110 M220]-[c(1)*M110(:,1)+c(2)*M110(:,2) c(1)*M110(:,1)+c(2)*M110(:,2) c(3)*M220(:,1)+c(4)*M220(:,2) c(3)*M220(:,1)+c(4)*M220(:,2)];
% elseif strcmp(centered,'all')
%     tstX_mean=[M110 M220]-[c(1)*M110(:,1)+c(2)*M110(:,2)+c(3)*M220(:,1)+c(4)*M220(:,2) c(1)*M110(:,1)+c(2)*M110(:,2)+c(3)*M220(:,1)+c(4)*M220(:,2) c(1)*M110(:,1)+c(2)*M110(:,2)+c(3)*M220(:,1)+c(4)*M220(:,2) c(1)*M110(:,1)+c(2)*M110(:,2)+c(3)*M220(:,1)+c(4)*M220(:,2)];
% end
%A=((2/lambda)*eye(2*p)+R*R')^(1/2)*((2/lambda)*eye(2*p)+R*R')^(1/2);
 
%Dtilde=inv((A^(1/2)*(Z*Z')*A^(1/2))+eye(k*p));
%test1=A*(Z/H);
%Dia=diag([(sqrt(gamma(1)))*ones(n1,1);(sqrt(gamma(2)))*ones(n2,1)]);
%Dia2=diag([(1/(gamma(1)))*ones(n1,1);(1/(gamma(2)))*ones(n2,1)]);
%test2=A^(1/2)*inv(A^(1/2)*Dia1*Z*Z'*Dia1*A^(1/2)+eye(p*k))*A^(1/2)*Dia1*Dia1*Z
param=struct();
param.gamma=gamma;param.lambda=lambda;param.nt=[n1;n2];
[out] = delta_F(p,param,C,c,co,m,'synthetic');
out_verif=[];
for task=1:k-1
    out_verif=[out_verif;out(m*(task-1)+1)];
end
%[out,Qtildez2] = delta_F(p,param,C,c,co,2,'synthetic','general');
% out=kron(out_verif,ones(m,1));
%Mdelta = bsxfun(@rdivide, Mb, (1+out)');
% Mdelta1 = bsxfun(@rdivide, Mb, (1+out)');
invQtilde=zeros(k*p,k*p);
for i=1:k
    for j=1:m
         invQtilde=invQtilde+(c(m*(i-1)+j)/co)*squeeze(C(:,:,m*(i-1)+j))/((1+out(m*(i-1)+j)));
%         invQtilde=invQtilde+(c(2*(i-1)+j)/co)*squeeze(C(:,:,2*(i-1)+j));
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
%  score_test=(1/(sqrt(k*p)))*tstX1'*A^(1/2)*Dtilde*A^(1/2)*Z*(trnY(:)-P*b);
% teste=(1/(k*p))*Mgota'*Qtildez*Mgota*(J'*(trnY(:)-P*b)./((1+out).*sqrt(are)))./(sqrt(are))+[b(1);b(1);b(2);b(2)];
%score_mean=(1/(k*p))*tstX_mean'*A^(1/2)*Qtildez*A^(1/2)*Mdelta1*J'*(trnY(:)-P*b)+[b(1);b(1);b(2);b(2)];
for task=1:k-1 
    X1{task}=trnXs{task};
end
 X1{k}=trnXt;
MM=zeros(2*k,2*k);
MMo=zeros(2*k,2*k);
for i=1:k
    ei=zeros(k,1);ei(i)=1;
    Delta_mui=(M(:,m*(i-1)+1)-M(:,m*(i-1)+2));
    ci1=c(2*(i-1)+1);ci2=c(2*(i-1)+2);
    ci=ci1+ci2;
    tildeD(i)=ci/(co*(1+out(2*(i-1)+1)));
    for j=1:k
        ej=zeros(k,1);ej(j)=1;
        cj1=c(2*(j-1)+1);cj2=c(2*(j-1)+2);
        cj=cj1+cj2;
        Delta_muj=(M(:,2*(j-1)+1)-M(:,2*(j-1)+2));
        if i~=j
            cardSi1=nt(2*(i-1)+1);cardSi2=nt(2*(i-1)+2);cardSj1=nt(2*(j-1)+1);cardSj2=nt(2*(j-1)+2);
            J_i1=zeros(nd(i),1);J_i2=zeros(nd(i),1);J_j1=zeros(nd(j),1);J_j2=zeros(nd(j),1);
            J_i1(1:cardSi1)=1;J_j1(1:cardSj1)=1;J_i2(cardSi1+1:end)=1;J_j2(cardSj1+1:end)=1;
            Deltamui_Deltamuj=(2*p)*(J_i1/cardSi1-J_i2/cardSi2)'*(X1{i}'*X1{j})*(J_j1/cardSj1-J_j2/cardSj2);
        else
            cardSi1=floor(nt(2*(i-1)+1)/2);cardSip1=nt(2*(i-1)+1)-cardSi1;
            cardSi2=floor(nt(2*(i-1)+2)/2);cardSip2=nt(2*(i-1)+2)-cardSi2;
            J_i1=zeros(nd(i),1);J_i2=zeros(nd(i),1);J_ip1=zeros(nd(i),1);J_ip2=zeros(nd(i),1);
            J_i1(1:cardSi1)=1;J_ip1(cardSi1+1:nt(2*(i-1)+1))=1;J_i2(nt(2*(i-1)+1)+1:nt(2*(i-1)+1)+cardSi2)=1;
            J_ip2(nt(2*(i-1)+1)+cardSi2+1:end)=1;
            Deltamui_Deltamuj=(2*p)*(J_i1/cardSi1-J_i2/cardSi2)'*(X1{i}'*X1{i})*(J_ip1/cardSip1-J_ip2/cardSip2);
        end
         MMo=MMo+kron(Delta_mui'*Delta_muj*ei*ej',sqrt(ci1*ci2/(ci^3))*[sqrt(ci2);-sqrt(ci1)]*[sqrt(cj2) -sqrt(cj1)]*sqrt(cj1*cj2/(cj^3)));
%         MM=MM+kron(Deltamui_Deltamuj*ei*ej',sqrt(ci1*ci2/(ci^3))*[sqrt(ci2);-sqrt(ci1)]*[sqrt(cj2) -sqrt(cj1)]*sqrt(cj1*cj2/(cj^3)));
        
    end
end
% Mgota_test=zeros(k*p,2*k);
% for i=1:k
%     ei=zeros(k,1);ei(i)=1;
%     for j=1:2
%         eij=zeros(2*k,1);eij(2*(i-1)+j)=1;
%         Mgota_test=Mgota_test+kron((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2)*ei*eij',sqrt(nt(2*(i-1)+j)/(k*p*(1+out(2*(i-1)+j))))*M(:,2*(i-1)+j));
%     end
% end
% Agotique=(eye(k)+diag(tildeD)^(-1/2)*(diag(gamma)+lambda*ones(k,1)*ones(1,k))^(-1)*diag(tildeD)^(-1/2))^(-1);
% MQ0M1=kron(Agotique,ones(2,1)*ones(1,2)).*MM;
are=c./(co*(1+out));
Mgota=A^(1/2)*M0*diag(sqrt(are));
% Mgotac1=A^(1/2)*M0c1*diag(sqrt(are));
% Mgotac2=A^(1/2)*M0c2*diag(sqrt(are));
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
% score_th2=ytilde-diag(deltabar)^(-1/2)*Gamma2*diag(deltabar)^(1/2)*ytilde0;
 
%  Zi=Z;Zi(:,1)=[];
% Dtildei=(A^(1/2)*Zi*Zi'*A^(1/2)+eye(k*p))^(-1);
%delta=Z(:,1)'*A^(1/2)*Dtildei*A^(1/2)*Z(:,1);
%ea=zeros(k,1);ea(1)=1;
%M_got_t=kron(((2/lambda*eye(k)+ones(k,1)*ones(1,k))^(1/2))*ea,M(:,1));
% Mgot=[];are=zeros(2*k,1);aren=zeros(k,1);
% for i=1:k
%     for j=1:2
%         ei=zeros(k,1);ei(i)=1;
%         Mgot=[Mgot kron(((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2))*ei,M(:,2*(i-1)+j))];
%         are(2*(i-1)+j)=(nt(2*(i-1)+j)/(k*p*(1+out(2*(i-1)+j))));
%     end
%     aren(i)=are(2*(i-1)+1)+are(2*(i-1)+2);
% end
% invQ2=zeros(k*p,k*p);
% for i=1:k
%     d=zeros(k,1);d(i)=1;%d=d*d';
%     %invQ2=invQ2+aren(i)*((kron(((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2)*d*((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2),eye(p))));
%     for j=1:2
%         %invQtilde=invQtilde+c(2*(i-1)+j)*squeeze(C(:,:,2*(i-1)+j))/(co*(1+out(2*(i-1)+j)));
%         invQ2=invQ2+sqrt(are(2*(i-1)+j))*kron(((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2)*diag(1./sqrt(gamma))*d,M(:,2*(i-1)+j))*sqrt(are(2*(i-1)+j))*kron(d'*((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2)*diag(1./sqrt(gamma)),M(:,2*(i-1)+j)');
%     end
% end
% invQ2=invQ2+((kron(((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2)*diag(aren)*((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2),eye(p))));
%Qtest2=inv(invQ2+(1/gamma)*eye(k*p));
% Mgota=bsxfun(@rdivide, Mgot, 1./sqrt(are)');
% for kj=1:2*k
%     Mgota(:,kj)=Mgot(:,kj)*sqrt(are(kj));
% end
% Mgota2=zeros(k*p,2*k);
% for i=1:k
%     ei=zeros(k,1);ei(i)=1;
%     Delta_mui=(M(:,2*(i-1)+1)-M(:,2*(i-1)+2));
%     ci1=c(2*(i-1)+1);ci2=c(2*(i-1)+2);
%     ci=ci1+ci2;
%     Mgota2=Mgota2+kron((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2)*ei*ei',kron([sqrt(ci1)*ci2/ci -sqrt(ci2)*ci1/ci],Delta_mui/sqrt(co*(1+out(2*(i-1)+1)))));
% end
%Qtest1=inv()
%INT=inv((((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2))*diag(aren)*(((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2))+(1/gamma)*eye(k));
%Qtilde0=kron(INT,eye(p));vr=1;br=1;
% invQtilde0=zeros(k*p,k*p);
% for g=1:k
%     e=zeros(k,1);e(g)=1;
%     invQtilde0=invQtilde0+kron(((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2)*e*e'*((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2))),are(2*(g-1)+1)*Ct(:,:,2*(g-1)+1)+are(2*(g-1)+2)*Ct(:,:,2*(g-1)+2));
% end
% invQtilde0=kron((((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2)*diag(1./sqrt(gamma))*diag(aren)*(((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2)*diag(1./sqrt(gamma)))),eye(p));
% Qtilde0=inv(invQtilde0+eye(k*p));
% Qtilde0=kron(inv((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2)*diag(aren)*((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2))+eye(k)),eye(p));
%inter=Mgota'*Qtilde0*Mgota;ei=zeros(k,1);ei(vr)=1;ej=zeros(k,1);ej(br)=1;
%Df=inv((2/lambda)*diag((aren.^2))+(1/gamma)*diag(aren));
%Df1=inv((2/lambda)*diag((aren))+(1/gamma)*eye(k));con=1/(1+ones(1,k)*diag(aren)*Df1*ones(k,1));
%rty=Df-Df1*ones(k,1)*ones(1,k)*Df1*con;
%inter_el=kron(ei'*(diag(1./aren)-(1/gamma)*Df)*ej,M(:,br)'*M(:,vr))*(sqrt(are(br)*are(vr)))+...
%    kron(ei'*((1/gamma)*Df1*ones(k,1)*ones(1,k)*Df1*con)*ej,M(:,br)'*M(:,vr))*(sqrt(are(br)*are(vr)));
%Qtest=Qtilde0-Qtilde0*Mgota*inv(eye(2*k)+Mgota'*Qtilde0*Mgota)*Mgota'*Qtilde0;
%v=nt'.*ytilde./(sqrt(are).*(1+out));
%v2=ytilde;
%score_test=(1/(k*p))*v'*(eye(2*k)-(eye(2*k)+MQ0M)^(-1))*diag(1./sqrt(c./(co*(1+out'))))
%vp=v;
%M_got_t=kron((((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2))*ea,M(:,1));
%score_test=(1/(sqrt(k*p)))*tstX3'*A^(1/2)*Dtilde*A^(1/2)*Z*(trnY(:)-P*b)+b(2);
%score3=(1/sqrt(2*p))*((2/lambda)*tstX3'*Z*alpha+tstX3'*(R*R')*Z*alpha)+ones(tstN,1)*b(2);
S1=zeros(k*p,k*p,m*k);d1=zeros(m*k,1);
for i=1:k
    for j=1:m
        d1(m*(i-1)+j)=nt(m*(i-1)+j)/(k*p*((1+out(m*(i-1)+j))^2));
        e=zeros(k,1);e(i)=1;e=e*e';
        S1(:,:,m*(i-1)+j)=kron(e,Ct(:,:,m*(i-1)+j));
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
for i=1:k
    for j=1:m
%         TS=zeros(k*p,k*p);
        TS2=zeros(k*p,k*p);
        for l=1:k
            for mc=1:m
%                 TS=TS+d1(2*(l-1)+m)*T(2*(i-1)+j,2*(l-1)+m)*Qtildez*squeeze(C(:,:,2*(l-1)+m))*Qtildez;
                kappa(m*(i-1)+j,m*(l-1)+mc)=d1(m*(l-1)+mc)*T(m*(i-1)+j,m*(l-1)+mc);
                TS2=TS2+kappa(m*(i-1)+j,m*(l-1)+mc)*A^(1/2)*S1(:,:,m*(l-1)+mc)*A^(1/2);
                
            end
        end
%         Tto(:,:,2*(i-1)+j)=TS;
        V(:,:,m*(i-1)+j)=A^(1/2)*S1(:,:,m*(i-1)+j)*A^(1/2)+TS2;
    end
end
% kappa1=[kappa(1,1) kappa(1,3);kappa(3,1) kappa(3,3)];
% Agotique2=(eye(k)+diag(tildeD)^(-1/2)*(diag(gamma)+lambda*ones(k,1)*ones(1,k))^(-1)*diag(tildeD)^(-1/2))^(-1);
% MGM1=zeros(m*k,m*k,k*p);
% MGM2=zeros(m*k,m*k,k*p);MGM3=zeros(m*k,m*k,k*p);
% for sd=1:k
%     for kd=1:m
%           ei=zeros(k,1);ei(sd)=1;
%          kappabar=[kappa(m*(sd-1)+kd,1)+kappa(m*(sd-1)+kd,2);kappa(m*(sd-1)+kd,3)+kappa(m*(sd-1)+kd,4)];
%          MGM1(:,:,m*(sd-1)+kd)=Mgota'*Qtilde0*V(:,:,m*(sd-1)+kd)*Qtilde0*Mgota;
%          V2(:,:,m*(sd-1)+kd)=kron((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2)*(ei*ei'+diag(kappabar))*(diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2),eye(p));
% %          MGM2(:,:,m*(sd-1)+kd)=Mgota'*Qtilde0*V2(:,:,m*(sd-1)+kd)*Qtilde0*Mgota;
% %          MGM3(:,:,m*(sd-1)+kd)=(kron(Agotique2*diag(tildeD)^(-1/2)*(ei*ei'+diag(kappabar))*diag(tildeD)^(-1/2)*Agotique2,ones(k,1)*ones(1,k))).*MMo;
%     end
% end
% TT2=(Agotique2.*Agotique2)/k;
% ei=zeros(k,1);ei(1)=1;ej=zeros(k,1);ej(1)=1;
% Agotique=diag(tildeD)^(-1/2)*(eye(k)+diag(tildeD)^(-1/2)*(diag(gamma)+lambda*ones(k,1)*ones(1,k))^(-1)*diag(tildeD)^(-1/2))^(-1)*diag(tildeD)^(-1/2);
% test2=Agotique.*Agotique/k
% D2=diag([n1/(k*p*((1+out(m*(i-1)+1))^2));n2/(k*p*((1+out(m*(i-1)+1))^2))])
% KAPPA=(D2*(Agotique.*Agotique/k))/(eye(k)-D2*(Agotique.*Agotique/k));
% for i=1:k
%     tildeD(i)=are(2*(i-1)+1);
% end
% ei=zeros(k,1);ei(k)=1;

% B=zeros(k*p,k*p,2*k);
% TB=zeros(k*p,k*p,2*k);
% for i=1:k
%     for j=1:2
%         B(:,:,2*(i-1)+j)=Qtildez*C(:,:,2*(i-1)+j)*Qtildez+Tto(:,:,2*(i-1)+j);
%         %B(:,:,2*(i-1)+j)=Dtilde*C(:,:,2*(i-1)+j)*Dtilde;
%     end
% end
% for i=1:k
%     for j=1:2
%         TI=zeros(k*p,k*p);TI2=zeros(k*p,k*p);
%         for l=1:k
%             for m=1:2
%                 TI=TI+d1(2*(l-1)+m)*trace(A^(1/2)*squeeze(S1(:,:,2*(i-1)+j))*A^(1/2)*Qtildez*squeeze(C(:,:,2*(l-1)+m))*Qtildez)*squeeze(B(:,:,2*(l-1)+m))/(k*p);
%             end
%         end
%         TB(:,:,2*(i-1)+j)=TI;
%     end
% end
% Ab=zeros(k*p,k*p,2*k);Cg=[];Ab1=zeros(k*p,k*p,2*k);
% for i=1:k
%     for j=1:2
%         %f=zeros(2*k,1);f(2*(i-1)+j)=1;
%         %Ab1(:,:,2*(i-1)+j)=Qtildez*A^(1/2)*S1(:,:,2*(i-1)+j)*A^(1/2)*Qtildez+TB(:,:,2*(i-1)+j);
%         Ab(:,:,2*(i-1)+j)=Qtildez*A^(1/2)*S1(:,:,2*(i-1)+j)*A^(1/2)*Qtildez+Tto(:,:,2*(i-1)+j);
% %         Ab(:,:,2*(i-1)+j)=Dtilde*A^(1/2)*S1(:,:,2*(i-1)+j)*A^(1/2)*Dtilde;
%         %jt=J(:,2*(i-1)+j);
%         %Cg=[Cg;jt'*trace(Ab(:,:,2*(i-1)+j)*A^(-1/2)*C(:,:,2*(i-1)+j))/(k*p*(1+(1/co)*out(2*(i-1)+j)))];
%         %vecC=[vecC;ones(nt(2*(i-1)+j),1)*trace(C(:,:,2*(i-1)+j)*Ab(:,:,2*(i-1)+j))/(1+out(2*(i-1)+j))];
%         %Ep=Ep+(1/(k*p))*trace(C(:,:,2*(i-1)+j)*Ab(:,:,2*(i-1)+j))*f*f';
%     end
% end
% kappa1=zeros(2*k,2*k);
% Ab3=zeros(k*p,k*p,2*k);
% out3=zeros(2*k,2*k);
% out3M=zeros(2*k,2*k);
% out4M=zeros(2*k,2*k);
% out5M=zeros(2*k,2*k);
% Delt=[out(1);out(3)];
% tildeD=[n1;n2]./((k*p)*(1+out_verif));
% Agotique1=inv(eye(k)+(diag(tildeD)^(-1/2)/(diag(gamma)+lambda*ones(k,1)*ones(1,k)))*diag(tildeD)^(-1/2));
% cb=[n1;n2]/n;
% DI=diag(co./(k*cb));
% KAPPA=(1/k)*((Agotique1.*Agotique1)^(1/2)/(eye(k)-(Agotique1.*Agotique1)^(1/2)*DI*(Agotique1.*Agotique1)^(1/2)))*(Agotique1.*Agotique1)^(1/2);
% for i=1:k
%     for j=1:2
%         Res(:,:,2*((i-1)+j))=zeros(k*p,k*p);
%         Res2(:,:,2*((i-1)+j))=zeros(k*p,k*p);
%         [out2] = kappa_F(S1(:,:,2*(i-1)+j),Qtildez,C,A,nt,m,p,out,param,'identity','synthetic');
%         [out3(:,2*(i-1)+j)] = kappa_F(S1(:,:,2*(i-1)+j),Qtildez,C,A,nt,m,p,out,param,'general','synthetic');
%         out3M(:,2*(i-1)+j)=(out3(:,2*(i-1)+j)/(p*k))*(tildeD(i)).*(kron(tildeD,ones(2,1)));
%         out3M=kron(KAPPA,ones(2,1)*ones(1,2));
%         out4M(:,2*(i-1)+j)=out3M(:,2*(i-1)+j)*(co)*(1./tildeD(i))./(cbar);
%         out5M(:,2*(i-1)+j)=out3M(:,2*(i-1)+j).*((1/tildeD(i))./(cbar)).*(c./(1+out));
%         % out24(:,2*(i-1)+j)=out3*(nt(2*(i-1)+j)/((k*p)^2*(1+out(2*(i-1)+j))^2));
%         for n1=1:k
%             for m1=1:2
%                 kappa1(2*(n1-1)+m1,2*(i-1)+j)=(nt(2*(n1-1)+m1)/((k*p)^2*(1+out(2*(n1-1)+m1))^2))*trace(C(:,:,2*(n1-1)+m1)*Dtilde*A^(1/2)*S1(:,:,2*(i-1)+j)*A^(1/2)*Dtilde);
%                 Res(:,:,(2*(i-1)+j))=Res(:,:,(2*(i-1)+j))+kappa1(2*(i-1)+j,2*(n1-1)+m1)*Qtildez*C(:,:,2*(n1-1)+m1)*Qtildez;
%                 Res2(:,:,(2*(i-1)+j))=Res2(:,:,(2*(i-1)+j))+(nt(2*(n1-1)+m1)/((k*p)^2*(1+out(2*(n1-1)+m1))^2))*trace(A^(1/2)*squeeze(S1(:,:,2*(i-1)+j))*A^(1/2)*Qtildez*squeeze(C(:,:,2*(l-1)+m))*Qtildez)*Dtilde*C(:,:,2*(n1-1)+m1)*Dtilde;
%             end
%         end
% %         Ab3(:,:,2*(i-1)+j)=Dtilde*A^(1/2)*S1(:,:,2*(i-1)+j)*A^(1/2)*Dtilde;
% %         Ab_est(:,:,2*(i-1)+j)=Qtildez*A^(1/2)*S1(:,:,2*(i-1)+j)*A^(1/2)*Qtildez+Res(:,:,2*(i-1)+j);
% %         Ab_est2(:,:,2*(i-1)+j)=Qtildez*A^(1/2)*S1(:,:,2*(i-1)+j)*A^(1/2)*Qtildez+Res2(:,:,2*(i-1)+j);
%     end
% end
% for i=1:k
%     for j=1:2
%         ;
%     end
% end
%  J=zeros(n,2*k);
% for i=1:2*k
%     J(sum(nt(1:i-1))+1:sum(nt(1:i)),i)=ones(nt(i),1);
% end
% for i=1:k
%     for j=1:2
%         VCg=[];VEp=[];VEp2=[];VCg2=[];
%         for l=1:k
%             for m=1:2
%                 f=zeros(2*k,1);f(2*(l-1)+m)=1;
%                 jt=J(:,2*(l-1)+m);
%                 VCg=[VCg;trace(Ab(:,:,2*(i-1)+j)*C(:,:,2*(l-1)+m))/((1+out(2*(l-1)+m)))];
%                 VCg2=[VCg2;jt'*trace(Ab(:,:,2*(i-1)+j)*C(:,:,2*(l-1)+m))/(k*p*(1+out(2*(l-1)+m)))];
%                 %vecC=[vecC;ones(nt(2*(i-1)+j),1)*trace(C(:,:,2*(i-1)+j)*Ab(:,:,2*(i-1)+j))/(1+out(2*(i-1)+j))];
%                 VEp=[VEp;(trace(C(:,:,2*(l-1)+m)*Ab(:,:,2*(i-1)+j))/((1+out(2*(l-1)+m)).^2))*ones(nt(2*(l-1)+m),1)];
%                 VEp2=[VEp2;(trace(C(:,:,2*(l-1)+m)*Ab(:,:,2*(i-1)+j))/(k*p*(1+out(2*(l-1)+m))))];
%             end
%         end
%         Cg(:,:,2*(i-1)+j)=diag(VCg);
%         Cg2(:,:,2*(i-1)+j)=VCg2;
%         Ep(:,:,2*(i-1)+j)=diag(VEp);
%         Ep2(:,:,2*(i-1)+j)=diag(VEp2);
%     end
% end
% Cg=diag(vecC)/(p);
% variance=zeros(2*k,1);
% Mdelta1=M0*diag(1./(1+out));
% for i=1:k
%     for j=1:2
%         variance(2*(i-1)+j)=(1/(k*p)^2)*(trnY(:)-P*b)'*(J*Mdelta1'*A^(1/2)*Ab(:,:,2*(i-1)+j)*A^(1/2)*Mdelta1*J'+Ep(:,:,2*(i-1)+j))*(trnY(:)-P*b)-...
%             (2/(k*p)^2)*(trnY(:)-P*b)'*J*Mdelta1'*A^(1/2)*Qtildez*A^(1/2)*Mdelta1*Cg2(:,:,2*(i-1)+j)*(trnY(:)-P*b);
%     end
% end
% dr1=(1/(k*p)^2)*(trnY(:)-P*b)'*Ep(:,:,2*(i-1)+j)*(trnY(:)-P*b)
% dr1_t=ytilde0'*diag(sqrt(are))*Ep2(:,:,2*(i-1)+j)*diag(sqrt(are))*ytilde0
% dr2=(2/(k*p)^2)*(trnY(:)-P*b)'*J*Mdelta1'*A^(1/2)*Qtildez*A^(1/2)*Mdelta1*Cg2(:,:,2*(i-1)+j)*(trnY(:)-P*b)
% dr2_t=2*ytilde0'*diag(sqrt(are))*(eye(2*k)-Gamma)*Ep2(:,:,2*(i-1)+j)*diag(sqrt(are))*ytilde0
% dr22_t=2*ytilde0'*diag(sqrt(are))*(eye(2*k)-Gamma)*diag(1./are)*diag(kappa(2*(i-1)+j,:))*diag(sqrt(are))*ytilde0
Mat=zeros(m*k,m*k,m*k);
for i=1:k
    for j=1:m
%         MGMc1=Mgotac1'*Qtilde0*Mgotac2;
        MGM=Mgota'*Qtilde0*V(:,:,m*(i-1)+j)*Qtilde0*Mgota;
%         MGM(1:2,1:2)=MGMc1(1:2,1:2);MGM(3:4,3:4)=MGMc1(3:4,3:4);
        Mat(:,:,m*(i-1)+j)=diag(sqrt(are))*Gamma*(MGM+diag(kappa(m*(i-1)+j,:)./are'))*Gamma*diag(sqrt(are));
        variance_th(m*(i-1)+j)=ytilde0'*Mat(:,:,m*(i-1)+j)*ytilde0;
    end
end
% var2=2;

%test1=(1/(k*p)^2)*(trnY(:)-P*b)'*(J*Mdelta1'*A^(1/2)*Ab(:,:,2*(i-1)+j)*A^(1/2)*Mdelta1*J')*(trnY(:)-P*b);
%test1=(1/(k*p)^2)*v'*(Mgota'*Ab(:,:,2*(i-1)+j)*Mgota)*v;
%C11=((2/lambda)*are(1)/((2*aren(1)/lambda)+1/gamma));C12=((2/lambda)*sqrt(are(1))*sqrt(are(2))/((2*aren(1)/lambda)+1/gamma));C22=((2/lambda)*are(2)/((2*aren(1)/lambda)+1/gamma));
% C31=((2/lambda)*are(3)/((2*aren(2)/lambda)+1/gamma));C32=((2/lambda)*sqrt(are(3))*sqrt(are(4))/((2*aren(2)/lambda)+1/gamma));C33=((2/lambda)*are(4)/((2*aren(2)/lambda)+1/gamma));
% Ml1=(1+C11*M(:,1)'*M(:,1))*(1+C22*M(:,2)'*M(:,2))-(C12*M(:,1)'*M(:,2))^2;
% Ml2=(1+C31*M(:,3)'*M(:,3))*(1+C33*M(:,4)'*M(:,4))-(C32*M(:,3)'*M(:,4))^2;
% Gamcomp=[1-(1+C22*M(:,2)'*M(:,2))/Ml1 C12*M(:,1)'*M(:,2)/Ml1 zeros(1,2);C12*M(:,1)'*M(:,2)/Ml1 1-(1+C11*M(:,1)'*M(:,1))/Ml1 zeros(1,2);...
%     zeros(1,2) 1-(1+C33*M(:,4)'*M(:,4))/Ml2 C32*M(:,3)'*M(:,4)/Ml2;zeros(1,2) C32*M(:,3)'*M(:,4)/Ml2 1-(1+C31*M(:,3)'*M(:,3))/Ml2];
% Mat_var=zeros(k*p,k*p,k*2);
% %y_op=zeros(2*k,2*k);
% rtr=zeros(2*k,2*k);
%         alpha=(1/aren(1))*(1-((2/lambda+1)*gamma*aren(2)+1)/((4*gamma^2*aren(1)*aren(2))/lambda^2+(4*gamma^2*aren(1)*aren(2))/lambda+gamma*(2/lambda+1)*(aren(1)+aren(2))+1));
% xi=(1/aren(2))*(1-((2/lambda+1)*gamma*aren(1)+1)/((4*gamma^2*aren(1)*aren(2))/lambda^2+(4*gamma^2*aren(1)*aren(2))/lambda+gamma*(2/lambda+1)*(aren(1)+aren(2))+1));
% beta=gamma/((4*gamma^2*aren(1)*aren(2))/lambda^2+(4*gamma^2*aren(1)*aren(2))/lambda+gamma*(2/lambda+1)*(aren(1)+aren(2))+1);
% SZ1=are(1)+are(2)*(nt(1)/nt(2))^2;SZ2=are(3)+are(4)*(nt(3)/nt(4))^2;
% lambda1=0.5*(alpha*SZ1*M(:,1)'*M(:,1)+xi*SZ2*M(:,3)'*M(:,3))+0.5*sqrt((alpha*SZ1*M(:,1)'*M(:,1)-xi*SZ2*M(:,3)'*M(:,3))^2+4*(beta*M(:,1)'*M(:,3))^2*SZ1*SZ2);
% lambda2=0.5*(alpha*SZ1*M(:,1)'*M(:,1)+xi*SZ2*M(:,3)'*M(:,3))-0.5*sqrt((alpha*SZ1*M(:,1)'*M(:,1)-xi*SZ2*M(:,3)'*M(:,3))^2+4*(beta*M(:,1)'*M(:,3))^2*SZ1*SZ2);
% a1=alpha*M(:,1)'*M(:,1)*are(1)-lambda2;b1=-(nt(1)/nt(2))*sqrt(are(1)*are(2))*alpha*M(:,1)'*M(:,1);c=beta*M(:,1)'*M(:,3)*sqrt(are(1)*are(3));d=-beta*M(:,1)'*M(:,3)*(nt(3)/nt(4))*sqrt(are(1)*are(4));
% e=beta*M(:,1)'*M(:,3)*sqrt(are(3)*are(1));f=(-nt(1)/nt(2))*beta*M(:,1)'*M(:,3)*sqrt(are(3)*are(2));g1=xi*M(:,3)'*M(:,3)*are(3)-lambda2;h=(-nt(3)/nt(4))*xi*M(:,3)'*M(:,3)*sqrt(are(3)*are(4));
% i1=-(nt(3)/nt(4))*beta*M(:,1)'*M(:,3)*sqrt(are(1)*are(4));j1=(nt(1)*nt(3)/(nt(2)*nt(4)))*beta*M(:,1)'*M(:,3)*sqrt(are(4)*are(2)); kc=-(nt(3)/nt(4))*xi*M(:,3)'*M(:,3)*sqrt(are(3)*are(4)); l1=xi*M(:,3)'*M(:,3)*are(4)*(nt(3)/nt(4))^2-lambda2;l2=xi*M(:,3)'*M(:,3)*are(4)*(nt(3)/nt(4))^2-lambda1;
% a2=alpha*M(:,1)'*M(:,1)*are(1)-lambda1;g2=xi*M(:,3)'*M(:,3)*are(3)-lambda1;
% if abs(beta*M(:,1)'*M(:,3))/(abs(alpha)*M(:,1)'*M(:,1))<1e-10
%     u1=[-b1/(sqrt(a1^2+b1^2));a1/(sqrt(a1^2+b1^2));0;0];
%     u2=[0;0;-h/(sqrt(g2^2+h^2));g2/(sqrt(g2^2+h^2))];
% else
%     u1=[1;(-g1*a1*l1+a1*h^2+e^2*l1-2*e*h*d+d^2*g1)/(b1*g1*l1-b1*h^2-e*f*l1+2*d*f*h-d*g1*j1);(a1*f*l1-a1*h*j1-b1*e*l1+b1*h*d)/(b1*g1*l1-b1*h^2-e*f*l1+2*d*f*h-d*g1*j1);(-a1*f*h+a1*g1*j1+b1*e*h-b1*g1*d)/(b1*g1*l1-b1*h^2-e*f*l1+2*d*f*h-d*g1*j1)]./norm([1;(-g1*a1*l1+a1*h^2+e^2*l1-2*e*h*d+d^2*g1)/(b1*g1*l1-b1*h^2-e*f*l1+2*d*f*h-d*g1*j1);(a1*f*l1-a1*h*j1-b1*e*l1+b1*h*d)/(b1*g1*l1-b1*h^2-e*f*l1+2*d*f*h-d*g1*j1);(-a1*f*h+a1*g1*j1+b1*e*h-b1*g1*d)/(b1*g1*l1-b1*h^2-e*f*l1+2*d*f*h-d*g1*j1)]);
%     u2=[1;(-g2*a2*l2+a2*h^2+e^2*l2-2*e*h*d+d^2*g2)/(b1*g2*l2-b1*h^2-e*f*l2+2*d*f*h-d*g2*j1);(a2*f*l2-a2*h*j1-b1*e*l2+b1*h*d)/(b1*g2*l2-b1*h^2-e*f*l2+2*d*f*h-d*g2*j1);(-a2*f*h+a2*g2*j1+b1*e*h-b1*g2*d)/(b1*g2*l2-b1*h^2-e*f*l2+2*d*f*h-d*g2*j1)]./norm([1;(-g2*a2*l2+a2*h^2+e^2*l2-2*e*h*d+d^2*g2)/(b1*g2*l2-b1*h^2-e*f*l2+2*d*f*h-d*g2*j1);(a2*f*l2-a2*h*j1-b1*e*l2+b1*h*d)/(b1*g2*l2-b1*h^2-e*f*l2+2*d*f*h-d*g2*j1);(-a2*f*h+a2*g2*j1+b1*e*h-b1*g2*d)/(b1*g2*l2-b1*h^2-e*f*l2+2*d*f*h-d*g2*j1)]);
% end
% U=[u1 u2];
% for i=1:k
%     for j=1:2
%         for l=1:k
%             for m=1:2
%                 T(2*(i-1)+j,2*(l-1)+m)=trace(C(:,:,2*(i-1)+j)*Dtilde*C(:,:,2*(l-1)+m)*Dtilde)/(k*p);
%             end
%         end
%     end
% end
%ia=1;ja=1;
% vz=(sqrt(c).*ytilde0./(sqrt((1+out))));
%vc=sqrt(kron(tildeD,ones(2,1))).*sqrt(c'*co./cbar).*ytilde
%eij=zeros(2*k,1);eij(2*(ia-1)+ja)=1;
%score_mean_test1=sqrt((1+out(2*(ia-1)+ja))./c(2*(ia-1)+ja))*vz'*(eye(2*k)-Gamma)*eij+[b(1) b(1) b(2) b(2)];
 
%score_mean_test=sqrt((1+out(2*(ia-1)+ja))/c(2*(ia-1)+j))*vz'*(eye(2*k)-(eye(2*k)+MQ0M)^(-1))*eij+b(ia);
%etilde=zeros(2*k,2*k);
% for i=1:k
%     for j=1:2
%         %N2=zeros(k*p,k*p);
%         %for l=1:k
%             %for m=1:2
%                 %kappa(2*(l-1)+m)=(d1(2*(l-1)+m)/(k*p))*trace(A^(1/2)*squeeze(S1(:,:,2*(i-1)+j))*A^(1/2)*Qtildez*squeeze(C(:,:,2*(l-1)+m))*Qtildez);
%                 %N3=zeros(k*p,k*p);
%                 %for ne=1:k
%                 %    for o=1:2
%                 %        zeta(2*(l-1)+m,2*(ne-1)+o)=d1(2*(ne-1)+o)*T(2*(l-1)+m,2*(ne-1)+o);
%                 %        N3=N3+zeta(2*(l-1)+m,2*(ne-1)+o)*squeeze(C(:,:,2*(ne-1)+o));
%                 %    end
%                 %end
%                 %N2=N2+kappa(2*(l-1)+m)*(C(:,:,2*(l-1)+m)+N3);
%                 %etilde(2*(l-1)+m,2*(i-1)+j)=trace(C(:,:,2*(l-1)+m)*Ab(:,:,2*(i-1)+j))/(k*p*(1+out(2*(l-1)+m)));
%                 %trace(C(:,:,2*(l-1)+m)*Qtildez*(A^(1/2)*S1(:,:,2*(i-1)+j)*A^(1/2)+rtr(1,2*(i-1)+j)*C(:,:,1)+rtr(2,2*(i-1)+j)*C(:,:,2)+rtr(3,2*(i-1)+j)*C(:,:,3)+rtr(4,2*(i-1)+j)*C(:,:,4))*Qtildez)/(k*p*(1+out(2*(l-1)+m)))
%                 %trace(C(:,:,2*(l-1)+m)*Qtildez*A^(1/2)*S1(:,:,2*(i-1)+j)*A^(1/2)*Qtildez)/(k*p*(1+out(2*(l-1)+m)))+...
%                 %(rtr(1,2*(i-1)+j)*trace(C(:,:,2*(l-1)+m)*Qtildez*C(:,:,1)*Qtildez)+rtr(2,2*(i-1)+j)*trace(C(:,:,2*(l-1)+m)*Qtildez*C(:,:,2)*Qtildez)+rtr(3,2*(i-1)+j)*trace(C(:,:,2*(l-1)+m)*Qtildez*C(:,:,3)*Qtildez)+rtr(4,2*(i-1)+j)*trace(C(:,:,2*(l-1)+m)*Qtildez*C(:,:,4)*Qtildez))/(k*p*(1+out(2*(l-1)+m)))  
%                 %vec3(2*(l-1)+m)=are(2*(l-1)+m).*trace(C(:,:,2*(l-1)+m)*Ab(:,:,2*(i-1)+j))./nt(2*(l-1)+m);
%         %    end
%         %end
%         %rtr(:,2*(i-1)+j)=(zeta+eye(2*k))'*kappa';
%         %N2_test=sum(bsxfun(@times,C,reshape((zeta+eye(2*k))'*kappa',1,1,4)),3);
%         %test=diag([rtr(1,2*(i-1)+j)/are(1);rtr(2,2*(i-1)+j)/are(2);rtr(3,2*(i-1)+j)/are(3);rtr(4,2*(i-1)+j)/are(4)]);
%         %test=diag(out4M(:,2*(i-1)+j));
%         %Mat_com(:,:,2*(i-1)+j)=A^(1/2)*S1(:,:,2*(i-1)+j)*A^(1/2)+[(rtr(1,2*(i-1)+j)+rtr(2,2*(i-1)+j))*(2/lambda)*eye(p) zeros(p);zeros(p) (2/lambda)*(rtr(3,2*(i-1)+j)+rtr(4,2*(i-1)+j))*eye(p)]+Mgota*test*Mgota';
% %         s1=(rtr(1,2*(i-1)+j)+rtr(2,2*(i-1)+j));
% %         s2=(rtr(3,2*(i-1)+j)+rtr(4,2*(i-1)+j));
% %         if i==1
% %             s1=s1+1;
% %         else
% %             s2=s2+1;
% %         end
%         %Mat_com=Gamma*Mgota'*Qtilde0*A^(1/2)*([s1*eye(p) zeros(p);zeros(p) s2*eye(p)])*A^(1/2)*Qtilde0*Mgota*Gamma+(eye(2*k)-Gamma)*test*(eye(2*k)-Gamma);
%         %Gamma*Mgota'*Qtilde0*[s1*eye(p) zeros(p);zeros(p) s2*eye(p)]*Qtilde0*Mgota*Gamma
%         %outp1=(eye(2*k)-Gamma)*test*(eye(2*k)-Gamma)+((eye(2*k)-Gamma).*([s1/((2/lambda)*aren(1)+1/gamma)*ones(k,k) zeros(k);zeros(k) s2/((2/lambda)*aren(2)+1/gamma)*ones(k,k)]))*Gamma;
%         %Mat_var(:,:,2*(i-1)+j)=A^(1/2)*S1(:,:,2*(i-1)+j)*A^(1/2)+N2;
%         %Mat_var2(:,:,2*(i-1)+j)=(eye(2*k)-Gamma)*diag(etilde(:,2*(i-1)+j));
%         %zti(:,2*(i-1)+j)=(nt'.*ytilde.*etilde(:,2*(i-1)+j)./((1+out).*sqrt(are)));
%         %Mat=(Gamma*(Mgota'*Qtilde0*Mat_var(:,:,2*(i-1)+j)*Qtilde0*Mgota)*Gamma-(Mat_var2(:,:,2*(i-1)+j)+Mat_var2(:,:,2*(i-1)+j)')+diag(etilde(:,2*(i-1)+j)));
%         %Mat=toeplitz(0.1.^(0:2*k-1));
%         %e1=zeros(2*k,1);e2=zeros(2*k,1);e1(3)=1./sqrt(are(3));e2(4)=1./sqrt(are(4));
%         %Matm=(e1-e2)'*(eye(2*k)-Gamma);
%         %opt_max=Mat\(Matm');
%         %variance_test1(2*(i-1)+j)=(1/(k*p)^2)*v'*Mat*v;
%         %y_op(:,2*(i-1)+j)=opt_max.*sqrt(are).*(1+out)./(nt');
%         %alpha1=alpha^2*s1+beta^2*s2;beta1=alpha*beta*s1+beta*xi*s2;xi1=beta^2*s1+xi^2*s2;
%         %lambda3inter=(0.5*(alpha1*SZ1*M(:,1)'*M(:,1)+xi1*SZ2*M(:,3)'*M(:,3))+0.5*sqrt((alpha1*SZ1*M(:,1)'*M(:,1)-xi1*SZ2*M(:,3)'*M(:,3))^2+4*(beta1*M(:,1)'*M(:,3))^2*SZ1*SZ2))*(1/(1+lambda1)^2);
%         %lambda4inter=(0.5*(alpha1*SZ1*M(:,1)'*M(:,1)+xi1*SZ2*M(:,3)'*M(:,3))-0.5*sqrt((alpha1*SZ1*M(:,1)'*M(:,1)-xi1*SZ2*M(:,3)'*M(:,3))^2+4*(beta1*M(:,1)'*M(:,3))^2*SZ1*SZ2))*(1/(1+lambda2)^2);
%         %variance2(2*(i-1)+j)=(1/(k*p)^2)*v'*Gamma^2*test*v+(1/(k*p)^2)*v'*(Gamma*Mgota'*Qtilde0*A^(1/2)*([s1*eye(p) zeros(p);zeros(p) s2*eye(p)])*A^(1/2)*Qtilde0*Mgota*Gamma)*v;
%         ez=zeros(k,1);ez(i)=1;
% %         cb=[cbar(1);cbar(3)];
%         %out22=[out(1);out(3)];
%         %aut=[out5M(1,2*(i-1)+j)+out5M(2,2*(i-1)+j);out5M(3,2*(i-1)+j)+out5M(4,2*(i-1)+j)];
%         %Vc_test=kron(Agotique*diag(aut+ez)*diag(tildeD)^(-1)*Agotique,ones(2,1)*ones(1,2)).*MM;
%         kappai=[kappa(2*(i-1)+j,1)+kappa(2*(i-1)+j,2);kappa(2*(i-1)+j,3)+kappa(2*(i-1)+j,4)];
%         Vc=kron(Agotique*(diag(kappai+ez))*diag(1./tildeD)*Agotique,ones(2,1)*ones(1,2)).*MM;
%         MGM=Mgota'*Qtilde0*V(:,:,2*(i-1)+j)*Qtilde0*Mgota;
% 
%         %Qtilde0=kron(inv((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2)*diag(tildeD)*((diag(gamma)+lambda*ones(k,1)*ones(1,k))^(1/2))+eye(k)),eye(p));
% %          variance_th2(2*(i-1)+j)=(1./(tildeD(i)))*((vz'*Gamma*(diag(kron(KAPPA(:,i),ones(2,1))./cbar)+Vc)*Gamma*vz));
%          variance_th2(2*(i-1)+j)=ytilde0'*diag(sqrt(are))*Gamma*(diag(kappa(2*(i-1)+j,:)./are')+Vc)*Gamma*diag(sqrt(are))*ytilde0;
%     end
% end
%score_mean_test1=sqrt((1+out(2*(ia-1)+ja))./c(2*(ia-1)+ja))*vz'*(eye(2*k)-Gamma)*eij+[b(1) b(1) b(2) b(2)];
% for i=1:k
%     ei=zeros(k,1);ei(i)=1;
%     kappa_s(i)=(nt(1)/(k*p*(1+out(1))^2))*trace(C(:,:,2*(i-1)+1)*Dtilde*A^(1/2)*kron(ei*ei',eye(p))*A^(1/2)*Dtilde)/(k*p);
%     Nu(:,:,i)=A^(1/2)*(kron(ei*ei',eye(p)))*A^(1/2)+kappa_s(i)*(C(:,:,2*(i-1)+1)+C(:,:,2*(i-1)+2));
% end
% tildeQ0=kron(inv((nt(1)/(p*(1+out(1))))*((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2)*diag(1./(gamma))*((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2)+eye(k)),eye(p))
% eo=zeros(4,1);eo(3)=1/sqrt(are(3));eo(4)=-1/sqrt(are(4));
% ve=v'*u2*u2'*eo;vr=v'*u1*u1'*eo;
% term1=(1/(k*p)^2)*v'*Gamma^2*test*v;
% term2=(1/(k*p)^2)*v'*U*diag([lambda4inter;lambda3inter])*U'*v;
% %Mgota'*Qtilde0*A^(1/2)*S1(:,:,1)*A^(1/2)*Qtilde0*Mgota+Mgota'*Qtilde0*A^(1/2)*rtrS1(:,:,1)*A^(1/2)*Qtilde0*Mgota
% %score_mean_test=((1/(k*p)).^2)*norm(Matm*opt_max).^2;
% 
% Lambda=diag([lambda2;lambda1]);
% mean_sim=(1/(2*p))*(v'*U*inv(eye(2)+Lambda^(-1))*U'*diag(sqrt(1./are)))+[b(1) b(1) b(2) b(2)];
% lambda1inter=1/(1+lambda1)^2;lambda2inter=1/(1+lambda2)^2;
% for i=1:2
%     for j=1:2
%         Lambda_vec=eig((Gamma*(Mgota'*Qtilde0*Mat_var(:,:,2*(i-1)+j)*Qtilde0*Mgota)*Gamma-(Mat_var2(:,:,2*(i-1)+j)+Mat_var2(:,:,2*(i-1)+j)')));
%         Lambda2=diag(Lambda_vec(abs(Lambda_vec)>1e-6));
%         %var_sim(2*(i-1)+j)=(1/(2*p)^2)*(v'*(U*Lambda2*U')*v+v'*diag(etilde(:,2*(i-1)+j))*v);
%         %variance3(2*(i-1)+j)=(1/(k*p)^2)*v'*(Gamma*(test+U*diag([lambda4inter;lambda3inter])*U')*Gamma)*v;
%     end
% end
% Cyp=diag(1./aren)*(eye(k)-inv(gamma*diag(aren)*((2/lambda)*eye(k)+ones(k,1)*ones(1,k))+eye(k)));
% Cyp2=(sqrt(are)*sqrt(are)').*(M'*M);
% ei=zeros(k,1);ek=zeros(k,1);ei(1)=1;ek(2)=1;
% ele=sqrt(are(1))*kron(ei'*((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2),M(:,1)')*Qtilde0*...
%   sqrt(are(4))*kron(((2/lambda)*eye(k)+ones(k,1)*ones(1,k))^(1/2)*ek,M(:,4))  ;
% 
% score_mean_test=(1/(k*p))*v'*(eye(2*k)-Gamma)*diag(sqrt(1./are))+[b(1) b(1) b(2) b(2)];
% test=kron(Diam,eye(2))*Gamma*kron(Diam,eye(2))^(-1)
% test2=inv(eye(2*k)+kron(Diam,eye(2))*(Mgota'*Qtilde0*Mgota)*kron(Diam,eye(2))^(-1))
% test3=inv(eye(2*k)+(Mgota'*Dia1*Qtilde0*Dia1^(-1)*Mgota))
% %score_mean_test2=(v2'*(eye(2*k)-Gamma))+[b(1) b(1) b(2) b(2)];
% score_mean2=(sqrt(are).*ytilde)'*(eye(2*k)-Gamma)*diag(sqrt(1./are))+[b(1) b(1) b(2) b(2)];
% term3=inv((nt(1)/(p*(1+out(1))))+(1/gamma)*((2/lambda)*eye(k)+ones(k,1)*ones(1,k)))*(nt(1)/(k*p*(1+out(1))))
% tr=inv(k*eye(k)+(1/gamma)*inv((2/lambda)*eye(2)+ones(k,1)*ones(1,k))*k*p*(1+out(1))/nt(1));
% c222=(p*(1+out(1)))/(k*(1+out(1))*(k*nt(1)*gamma+p*(1+out(1))*lambda)*(2*k*nt(1)*gamma+p*(1+out(1))*lambda))
% m1=k+p*(1+out(1))*lambda/(gamma*nt(1))
% m2=lambda^2*k*p*(1+out(1))/(gamma*nt(1)*k^2*(1+lambda))
% c111=1/(k+(p*(1+out(1))*lambda)/(gamma*nt(1)));
% c112=(gamma*lambda^2*nt(1)*p*(out(1) + 1))/(k*(lambda*p + gamma*k*nt(1) + out(1)*lambda*p)*(lambda*p + gamma*k*nt(1) + out(1)*lambda*p + gamma*k*lambda*nt(1)))
% %lambdaf=lambda;
% % gammaf=gamma;
% % fun=@(lambdaf)  (1/aren(1)).*(1-((2./lambdaf+1).*gammaf.*aren(2)+1)/((4*gammaf.^2*aren(1)*aren(2))./lambdaf.^2+(4*gammaf.^2*aren(1).*aren(2))./lambdaf+gammaf*(2./lambdaf+1).*(aren(1)+aren(2))+1));
% % lam=1e-3:0.1:10;
% % for i=1:length(lam)
% %     lam_vec(i)=fun(lam(i));
% % end
% % plot(lam,lam_vec)
% %y_opt=0;error_opt=0;
  figure
  for ht=1:m
   x{ht} = score_th(m*(k-1)+ht)+sqrt(variance_th(m*(k-1)+ht))*[-3:.1:3];
   y{ht} = normpdf(x{ht},score_th(m*(k-1)+ht),sqrt(variance_th(m*(k-1)+ht)));
   hold all
   plot(x{ht},y{ht}./sum(y{ht}),'LineWidth',3);
  end
    hold on
  histogram(real(score1(1:n_test(1),1)),80,'Normalization','probability')
  for i=1:m-1
    histogram(real(score1(sum(n_test(1:i))+1:sum(n_test(1:i+1)),1)),80,'Normalization','probability')
  end
  Ze=zeros(m*k,m*k); 
 for task=1:k-1
     rg_k=1+m*(task-1):2+m*(task-1);
     Ze(rg_k,rg_k)=[(1-nt(m*(task-1)+1)/n11{task}) -nt(m*(task-1)+2)/n11{task};-nt(m*(task-1)+1)/n11{task} (1-nt(m*(task-1)+2)/n11{task})];
     %Ze=[Ze;(1-nt(m*(task-1)+1)/n11{task}) -nt(m*(task-1)+2)/n11{task} 0 0;-nt(m*(task-1)+1)/n11{task} (1-nt(m*(task-1)+2)/n11{task}) 0 0];
 end
 rg_k=1+m*(k-1):2+m*(k-1);
 Ze(rg_k,rg_k)=[(1-nt(m*(k-1)+1)/n2) -nt(m*(k-1)+2)/n2; (-nt(m*(k-1)+1)/n2) (1-nt(m*(k-1)+2)/n2)];
e1=zeros(m*k,1);e2=zeros(m*k,1);e1(m*(k-1)+1)=1;e2(m*(k-1)+2)=1;
Matm=(e1-e2)'*(eye(m*k)-diag(deltabar)^(-1/2)*Gamma*diag(deltabar)^(1/2));
% m1=(e1)'*(eye(m*k)-diag(deltabar)^(-1/2)*Gamma*diag(deltabar)^(1/2));
% m2=(e2)'*(eye(m*k)-diag(deltabar)^(-1/2)*Gamma*diag(deltabar)^(1/2));
 x0=[];
 for task=1:k
     x0=[x0;1;-1];
 end
 %x0=Ze*x0;
%  Zp=[zeros(1,2) nt(m*(k-1)+1:m*(k-1)+m)'/sum(nt(m*(k-1)+1:m*(k-1)+m))];
% zeta=(-(m1*Ze*x0+Zp*x0)*(x0'*Ze'*Mat(:,:,m*(k-1)+2)*Ze*x0)+(m2*Ze*x0+Zp*x0)*(x0'*Ze'*Mat(:,:,m*(k-1)+1)*Ze*x0)+sqrt(x0'*Ze'*Mat(:,:,m*(k-1)+1)*Ze*x0)*sqrt(x0'*Ze'*Mat(:,:,m*(k-1)+2)*Ze*x0)*sqrt(log((x0'*Ze'*Mat(:,:,m*(k-1)+1)*Ze*x0)./(x0'*Ze'*Mat(:,:,m*(k-1)+2)*Ze*x0))*((x0'*Ze'*Mat(:,:,m*(k-1)+1)*Ze*x0)-(x0'*Ze'*Mat(:,:,m*(k-1)+2)*Ze*x0))+(m1*Ze*x0-m2*Ze*x0)^2))./...
%     (x0'*Ze'*Mat(:,:,m*(k-1)+1)*Ze*x0-x0'*Ze'*Mat(:,:,m*(k-1)+2)*Ze*x0);
% zeta=(-(score_th(3))*(variance_th(4))+(score_th(4))*(variance_th(3))+sqrt(variance_th(3))*sqrt(variance_th(4))*sqrt(log((variance_th(3))./(variance_th(4)))*((variance_th(3))-(variance_th(4)))+(score_th(3)-score_th(4))^2))./...
%     (variance_th(3)-variance_th(4));
x0=Ze*x0;
% orn=((sqrt(x01'*Mat(:,:,m*(k-1)+1)*x01)*(m2*x01-m1*x01))+(sqrt(x01'*Mat(:,:,m*(k-1)+1)*x01))*sqrt(2*log((x01'*Mat(:,:,m*(k-1)+1)*x01)./(x01'*Mat(:,:,m*(k-1)+2)*x01))*((x01'*Mat(:,:,m*(k-1)+1)*x01)-(x01'*Mat(:,:,m*(k-1)+2)*x01))+(m1*x01-m2*x01)^2))^2./...
%     (sqrt(x01'*Mat(:,:,m*(k-1)+2)*x01).*(x01'*Mat(:,:,m*(k-1)+1)*x01-(x01'*Mat(:,:,m*(k-1)+2)*x01))^2);
% orn_test=(zeta-score_th(4))^2./variance_th(4)
%Mat=(1./(tildeD(i)))*Gamma*(diag(kron(KAPPA(:,i),ones(2,1))./cbar)+Vc)*Gamma;
obj=@(vt) -((Matm*vt)^2)./(8*vt'*Mat(:,:,m*(k-1)+2)*vt);
% obj3=@(vt) -((sqrt(vt'*Mat(:,:,m*(k-1)+1)*vt)*(m2*vt-m1*vt))-(sqrt(vt'*Mat(:,:,m*(k-1)+1)*vt))*sqrt(2*log((vt'*Mat(:,:,m*(k-1)+1)*vt)./(vt'*Mat(:,:,m*(k-1)+2)*vt))*((vt'*Mat(:,:,m*(k-1)+1)*vt)-(vt'*Mat(:,:,m*(k-1)+2)*vt))+(m1*vt-m2*vt)^2))./...
%     (sqrt(vt'*Mat(:,:,m*(k-1)+2)*vt).*(vt'*Mat(:,:,m*(k-1)+1)*vt-(vt'*Mat(:,:,m*(k-1)+2)*vt))^2);
obj2=@(vt) 0.5*erfc(Matm*vt/(2*sqrt(2)*sqrt(vt'*Mat(:,:,m*(k-1)+1)*vt)))+0.5*erfc(Matm*vt/(2*sqrt(2)*sqrt(vt'*Mat(:,:,m*(k-1)+2)*vt)));
 [obj1,error_opt]=fminunc(obj2,x0);

 y_opt=pinv(Ze)*(obj1);
 y_opt=y_opt./norm(y_opt);
%  error_opt2=0.5*erfc(real(sqrt(abs((Matm/Mat(:,:,4))*Matm'/8))));
%inr=((score_mean_test)/(4*2*(abs(variance_test1(3)))));
 %plot(x1,y1./sum(y1),'r');plot(x2,y2./sum(y2),'b');plot(x3,y3./sum(y3),'g');plot(x4,y4./sum(y4),'c')
 in1=((score_th(1)-score_th(2))/(2*sqrt(2)*sqrt(abs(variance_th(1)))));
 in2=((score_th(1)-score_th(2))/(2*sqrt(2)*sqrt(abs(variance_th(2)))));
 in3=((score_th(1+m*(k-1))-score_th(2+m*(k-1)))/(2*sqrt(2)*sqrt(abs(variance_th(1+m*(k-1))))));
 in4=((score_th(1+m*(k-1))-score_th(2+m*(k-1)))/(2*sqrt(2)*sqrt(abs(variance_th(2+m*(k-1))))));
 error_th(1)=0.5*erfc(real(in1));
 error_th(2)=0.5*erfc(real(in2));
 error_th(3)=0.5*erfc(real(in3));
 error_th(4)=0.5*erfc(real(in4));
 pred1=zeros(size(tstX1,2),1);
 %moy=(mean(score1(1:n_test(1)))+mean(score1(n_test(1)+1:sum(n_test))))/2;
 moy=(score_th(3)+score_th(4))/2;
 pred1(score1>moy)=1;pred1(score1<moy)=-1;
%  pred2(score2>((mean(score1)+mean(score2))/2))=1;pred2(score2<((mean(score1)+mean(score2))/2))=-1;
%  pred3(score3>((mean(score3)+mean(score4))/2))=1;pred3(score3<((mean(score3)+mean(score4))/2))=-1;
%  pred4(score4>((mean(score3)+mean(score4))/2))=1;pred4(score4<((mean(score3)+mean(score4))/2))=-1;
 error_emp(1)=sum(pred1(1:n_test(1))~=ones(n_test(1),1))/n_test(1);
 error_emp(2)=sum(pred1(n_test(1)+1:sum(n_test))~=-ones(n_test(2),1))/n_test(2);
%  histogram(score1(1:n_test(1)))
%  hold on
%  histogram(score1(1+n_test(1):n_test(1)+n_test(2)))
%    error_emp(3)=sum(pred1(2*n_test(1)+1:3*n_test)~=-ones(n_test(2),1))/n_test(2);
   
% 
% if p==2
%     for i=1:2
%     ei=zeros(1,2);ei(i)=1;
%     w0{i}=kron(ei,eye(p))*((2/lambda)*eye(2*p))*Z*alpha2;
%     v4{i}=kron(ei,eye(p))*(kron(ones(2,1)*ones(1,2),eye(p)))*Z*alpha2;
%     w{i}=w0{i}+v4{i};
%     end
% droite1=@(x) -(w0{1}(1)/w0{1}(2))*x-(b(2)/w0{1}(2));
% droite2=@(x) -(w{1}(1)/w{1}(2))*x-(b(2)/w{1}(2));
% droite3=@(x) -(w0{2}(1)/w0{2}(2))*x-(b(2)/w0{2}(2));
% droite4=@(x) -(w{2}(1)/w{2}(2))*x-(b(2)/w{2}(2));
% xe=linspace(min([min(trnXs(1,:));min(trnXt(1,:))]),max([max(trnXs(1,:));max(trnXt(1,:))]),100);
% ymin=min([min(trnXs(2,:));min(trnXt(2,:))]);ymax=max([max(trnXs(2,:));max(trnXt(2,:))]);
%     figure
%     hold on
%     plot(trnXs(1,trnY(1:n1)==1),trnXs(2,trnY(1:n1)==1),'g*')
%     plot(trnXt(1,trnY(n1+1:end)==1),trnXt(2,trnY(n1+1:end)==1),'r*')
%     plot(trnXs(1,trnY(1:n1)==-1),trnXs(2,trnY(1:n1)==-1),'go')
%     plot(trnXt(1,trnY(n1+1:end)==-1),trnXt(2,trnY(n1+1:end)==-1),'ro')
%     plot(xe,droite1(xe),'b');plot(xe,droite2(xe),'g');plot(xe,droite3(xe),'k');plot(xe,droite4(xe),'r')
%     axis([min(xe),max(xe),ymin,ymax])
% end
end
