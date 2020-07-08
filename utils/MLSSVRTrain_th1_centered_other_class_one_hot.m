function [score1,y_opt,proba_the,error_emp] = MLSSVRTrain_th1_centered_other_class_one_hot(trnXs,trnXt,trnY, gamma, lambda,M,Ct,tstX1,nt,centered,k,n_test)
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
nu = H \ trnY; 
 
S = P'*eta;
b = (S\eta')*trnY;
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
% score2_t=(1/(sqrt(2*p)))*tstX1'*A*Z*(H\(trnY-P*b))+ones(tstN,1)*b(1);
% score1_t=(1/(sqrt(2*p)))*tstX1'*A^(1/2)*Dtilde*A^(1/2)*Z*(trnY-P*b)+ones(tstN,1)*b(1);
%score2=(1/sqrt(2*p))*((2/lambda)*tstX2'*Z*alpha2+tstX2'*(R*R')*Z*alpha2)+ones(tstN2,1)*b(1);
%score3=(1/sqrt(2*p))*((2/lambda)*tstX3'*Z*alpha2+tstX3'*(R*R')*Z*alpha2)+ones(tstN3,1)*b(2);
%score4=(1/sqrt(2*p))*((2/lambda)*tstX4'*Z*alpha2+tstX4'*(R*R')*Z*alpha2)+ones(tstN4,1)*b(2);
score_emp=zeros(m,m);
score_emp(1,:)=mean(score1(1:n_test(1),:));
for i=1:m-1
    score_emp(i+1,:)=mean(score1(1+sum(n_test(1:i)):sum(n_test(1:i+1)),:));
end
%score_emp(2)=mean(score2);score_emp(3)=mean(score3);score_emp(4)=mean(score4);
var_emp(:,:,1)=cov(score1(1:n_test(1),:));
for i=1:m-1
    var_emp(:,:,i+1)=cov(score1(1+sum(n_test(1:i)):sum(n_test(1:i+1)),:));
end
%var_emp(2)=var(score2);var_emp(3)=var(score3);var_emp(4)=var(score4);
%var_test=(1/(k*p))*(trnY-P*b)'*Z'*A^(1/2)*Ab(:,:,4)*A^(1/2)*Z*(trnY-P*b)
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
%  Mdelta1 = bsxfun(@rdivide, Mb, (1+out)');
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
%test11=tstX1(:,1)'*A*(Z/H)*(trnY-P*b);
%test12=tstX1(:,1)'*A*(Z/H)*(trnY-P*b)-(1/gamma)*tstX1(:,1)'*A*(Z/H)*P2*(inv(eye(k)-P2'*P2+(1/gamma)*(P2'/H)*P2)*(P2'/H))*(trnY-P*b);
%Dtilde_comp=inv(inv(Dtilde)-A^(1/2)*Z*P*Di*P'*Z'*A^(1/2));
%Dtilde_comp=Dtilde+(Dtilde*A^(1/2)*Z*P*Di^(1/2))*((eye(k)-Di^(1/2)*P'*Z'*A^(1/2)*Dtilde*A^(1/2)*Z*P*Di^(1/2))\(Di^(1/2)*P'*Z'*A^(1/2)*Dtilde));
%Dtilde_comp=Dtilde+(Dtilde*A^(1/2)*Z*P*Di^(1/2))*((eye(k)-Di^(1/2)*P'*P*Di^(1/2)+(1/gamma)*(Di^(1/2)*P'/H)*P*Di^(1/2))\(Di^(1/2)*P'*Z'*A^(1/2)*Dtilde));
%Eq_mat=P'*Z'*A^(1/2)*Dtilde*A^(1/2)*Z;
%Eq_mat=(1/(k*p))*P'*J*Mdelta1'*A^(1/2)*Qtildez*A^(1/2)*Mdelta1*J'+P'*diag(eq_mat);
%Equival=A^(1/2)*Qtildez*A^(1/2)*Mdelta1*J'+A^(1/2)*(Qtildez*A^(1/2)*Mdelta1*J'*P)*((eye(k)-P'*P+(1/gamma)*(P'/H)*P)\(Eq_mat));
%Equival2=A^(1/2)*Dtilde1*A^(1/2)*Z*P1;
%Equival=A^(1/2)*Qtildez*A^(1/2)*Mdelta1*J'*P1/sqrt(k*p)+(1/sqrt(k*p))*A^(1/2)*(Qtildez*A^(1/2)*Mdelta1*J'*P*Di^(1/2))*((eye(k)-Di^(1/2)*P'*P*Di^(1/2)+(1/gamma)*(Di^(1/2)*P'/H)*P*Di^(1/2))\(Di^(1/2)*Eq_mat))*P1;
% score11=tstX1'*Equival*(trnY-P*b)/sqrt(k*p);
% score12=tstX1'*Equival2*(trnY-P*b)/(sqrt(k*p));
%ones(1,k*p)*Qtildez1*ones(k*p,1)/(k*p)
%  score_test=(1/(sqrt(k*p)))*tstX1'*A^(1/2)*Dtilde*A^(1/2)*Z*(trnY-P*b);
% teste=(1/(k*p))*Mgota'*Qtildez*Mgota*(J'*(trnY-P*b)./((1+out).*sqrt(are)))./(sqrt(are))+[b(1);b(1);b(2);b(2)];
% score_mean=(1/(k*p))*tstX_mean'*A^(1/2)*Qtildez*A^(1/2)*Mdelta1*J'*(trnY-P*b)+[b(1);b(1);b(2);b(2)];
% for task=1:k-1 
%     X1{task}=trnXs{task};
% end
%  X1{k}=trnXt;
% MM=zeros(2*k,2*k);
% MMo=zeros(2*k,2*k);
% for i=1:k
%     ei=zeros(k,1);ei(i)=1;
%     Delta_mui=(M(:,m*(i-1)+1)-M(:,m*(i-1)+2));
%     ci1=c(2*(i-1)+1);ci2=c(2*(i-1)+2);
%     ci=ci1+ci2;
%     tildeD(i)=ci/(co*(1+out(2*(i-1)+1)));
%     for j=1:k
%         ej=zeros(k,1);ej(j)=1;
%         cj1=c(2*(j-1)+1);cj2=c(2*(j-1)+2);
%         cj=cj1+cj2;
%         Delta_muj=(M(:,2*(j-1)+1)-M(:,2*(j-1)+2));
%         if i~=j
%             cardSi1=nt(2*(i-1)+1);cardSi2=nt(2*(i-1)+2);cardSj1=nt(2*(j-1)+1);cardSj2=nt(2*(j-1)+2);
%             J_i1=zeros(nd(i),1);J_i2=zeros(nd(i),1);J_j1=zeros(nd(j),1);J_j2=zeros(nd(j),1);
%             J_i1(1:cardSi1)=1;J_j1(1:cardSj1)=1;J_i2(cardSi1+1:end)=1;J_j2(cardSj1+1:end)=1;
%             Deltamui_Deltamuj=(2*p)*(J_i1/cardSi1-J_i2/cardSi2)'*(X1{i}'*X1{j})*(J_j1/cardSj1-J_j2/cardSj2);
%         else
%             cardSi1=floor(nt(2*(i-1)+1)/2);cardSip1=nt(2*(i-1)+1)-cardSi1;
%             cardSi2=floor(nt(2*(i-1)+2)/2);cardSip2=nt(2*(i-1)+2)-cardSi2;
%             J_i1=zeros(nd(i),1);J_i2=zeros(nd(i),1);J_ip1=zeros(nd(i),1);J_ip2=zeros(nd(i),1);
%             J_i1(1:cardSi1)=1;J_ip1(cardSi1+1:nt(2*(i-1)+1))=1;J_i2(nt(2*(i-1)+1)+1:nt(2*(i-1)+1)+cardSi2)=1;
%             J_ip2(nt(2*(i-1)+1)+cardSi2+1:end)=1;
%             Deltamui_Deltamuj=(2*p)*(J_i1/cardSi1-J_i2/cardSi2)'*(X1{i}'*X1{i})*(J_ip1/cardSip1-J_ip2/cardSip2);
%         end
% %         MMo=MMo+kron(Delta_mui'*Delta_muj*ei*ej',sqrt(ci1*ci2/(ci^3))*[sqrt(ci2);-sqrt(ci1)]*[sqrt(cj2) -sqrt(cj1)]*sqrt(cj1*cj2/(cj^3)));
%         MM=MM+kron(Deltamui_Deltamuj*ei*ej',sqrt(ci1*ci2/(ci^3))*[sqrt(ci2);-sqrt(ci1)]*[sqrt(cj2) -sqrt(cj1)]*sqrt(cj1*cj2/(cj^3)));
%         
%     end
% end
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
ver=(trnY-P*b);ver2=(trnY);
pos=1;ytilde0=zeros(m*k,m);ytilde0(1,:)=ver(pos,:);
pos2=1;ytilde=zeros(m*k,m);ytilde(1,:)=ver2(pos2,:);
for i=1:m*k-1
    pos=pos+nt(i);
    ytilde0(i+1,:)=ver(pos,:);
    pos2=pos2+nt(i);
    ytilde(i+1,:)=ver2(pos2,:);
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
%score_test=(1/(sqrt(k*p)))*tstX3'*A^(1/2)*Dtilde*A^(1/2)*Z*(trnY-P*b)+b(2);
%score3=(1/sqrt(2*p))*((2/lambda)*tstX3'*Z*alpha+tstX3'*(R*R')*Z*alpha)+ones(tstN,1)*b(2);
S1=zeros(k*p,k*p,m*k);d1=zeros(m*k,1);
for i=1:k
    for j=1:m
        d1(m*(i-1)+j)=nt(m*(i-1)+j)/(k*p*((1+out(m*(i-1)+j))^2));
        e=zeros(k,1);e(i)=1;e=e*e';
        S1(:,:,m*(i-1)+j)=kron(e,Ct(:,:,m*(i-1)+j));
    end
end
%var1=(1/(2*p))*((trnY-P*b)'/H)*Z'*A*S1(:,:,1)*A*Z*(H\(trnY-P*b));
%var11=(1/(2*p))*((trnY-P*b)'/H)*P1*Z'*A*S1(:,:,1)*A*Z*P1*(H\(trnY-P*b));
%var12=(1/(2*p))*((trnY-P*b)'/H)*P2*Z'*A*S1(:,:,1)*A*Z*P2*(H\(trnY-P*b));
%variance_centred=(1/(k*p))*(trnY-P*b)'*P1*Z'*A^(1/2)*Dtilde1*A^(1/2)*S1(:,:,1)*A^(1/2)*Dtilde1*A^(1/2)*Z*P1*(trnY-P*b);
%variance_centred1=(1/(k*p))*(trnY-P*b)'*P1*Z'*A^(1/2)*Dtilde*A^(1/2)*S1(:,:,1)*A^(1/2)*Dtilde*A^(1/2)*Z*P1*(trnY-P*b)+...
%    (1/(k*p))*(trnY-P*b)'*P1*Z'*A^(1/2)*Dtilde*A^(1/2)*S1(:,:,1)*A^(1/2)*(Dtilde*A^(1/2)*Z*P*Di^(1/2))*((eye(k)-Di^(1/2)*P'*P*Di^(1/2)+(1/gamma)*(Di^(1/2)*P'/H)*P*Di^(1/2))\(Di^(1/2)*P'*Z'*A^(1/2)*Dtilde))*A^(1/2)*Z*P1*(trnY-P*b)+...
%    (1/(k*p))*(trnY-P*b)'*P1*Z'*A^(1/2)*(Dtilde*A^(1/2)*Z*P*Di^(1/2))*((eye(k)-Di^(1/2)*P'*P*Di^(1/2)+(1/gamma)*(Di^(1/2)*P'/H)*P*Di^(1/2))\(Di^(1/2)*P'*Z'*A^(1/2)*Dtilde))*A^(1/2)*S1(:,:,1)*A^(1/2)*Dtilde*A^(1/2)*Z*P1*(trnY-P*b)+...
%    (1/(k*p))*(trnY-P*b)'*P1*Z'*A^(1/2)*(Dtilde*A^(1/2)*Z*P*Di^(1/2))*((eye(k)-Di^(1/2)*P'*P*Di^(1/2)+(1/gamma)*(Di^(1/2)*P'/H)*P*Di^(1/2))\(Di^(1/2)*P'*Z'*A^(1/2)*Dtilde))*A^(1/2)*S1(:,:,1)*A^(1/2)*(Dtilde*A^(1/2)*Z*P*Di^(1/2))*((eye(k)-Di^(1/2)*P'*P*Di^(1/2)+(1/gamma)*(Di^(1/2)*P'/H)*P*Di^(1/2))\(Di^(1/2)*P'*Z'*A^(1/2)*Dtilde))*A^(1/2)*Z*P1*(trnY-P*b);
 
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
%         TS=zeros(k*p,k*p);
        TS2=zeros(k*p,k*p);
        for l=1:k
            for mc=1:m
%                 TS=TS+d1(2*(l-1)+m)*T(2*(i-1)+j,2*(l-1)+m)*Qtildez*squeeze(C(:,:,2*(l-1)+m))*Qtildez;
                kappa(m*(i-1)+j,m*(l-1)+mc)=d1(m*(l-1)+mc)*T(m*(i-1)+j,m*(l-1)+mc);
                TS2=TS2+kappa(m*(i-1)+j,m*(l-1)+mc)*Rep(:,:,m*(l-1)+mc);
                
            end
        end
%         Tto(:,:,2*(i-1)+j)=TS;
        V(:,:,m*(i-1)+j)=Rep(:,:,m*(i-1)+j)+TS2;
    end
end
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
%         variance(2*(i-1)+j)=(1/(k*p)^2)*(trnY-P*b)'*(J*Mdelta1'*A^(1/2)*Ab(:,:,2*(i-1)+j)*A^(1/2)*Mdelta1*J'+Ep(:,:,2*(i-1)+j))*(trnY-P*b)-...
%             (2/(k*p)^2)*(trnY-P*b)'*J*Mdelta1'*A^(1/2)*Qtildez*A^(1/2)*Mdelta1*Cg2(:,:,2*(i-1)+j)*(trnY-P*b);
%     end
% end
% dr1=(1/(k*p)^2)*(trnY-P*b)'*Ep(:,:,2*(i-1)+j)*(trnY-P*b)
% dr1_t=ytilde0'*diag(sqrt(are))*Ep2(:,:,2*(i-1)+j)*diag(sqrt(are))*ytilde0
% dr2=(2/(k*p)^2)*(trnY-P*b)'*J*Mdelta1'*A^(1/2)*Qtildez*A^(1/2)*Mdelta1*Cg2(:,:,2*(i-1)+j)*(trnY-P*b)
% dr2_t=2*ytilde0'*diag(sqrt(are))*(eye(2*k)-Gamma)*Ep2(:,:,2*(i-1)+j)*diag(sqrt(are))*ytilde0
% dr22_t=2*ytilde0'*diag(sqrt(are))*(eye(2*k)-Gamma)*diag(1./are)*diag(kappa(2*(i-1)+j,:))*diag(sqrt(are))*ytilde0
Mat=zeros(m*k,m*k,m*k);
% variance_th=zeros(m,m,m*k);
for i=1:k
    for j=1:m
%         MGMc1=Mgotac1'*Qtilde0*Mgotac2;
        MGM=Mgota'*Qtilde0*V(:,:,m*(i-1)+j)*Qtilde0*Mgota;
%         MGM(1:2,1:2)=MGMc1(1:2,1:2);MGM(3:4,3:4)=MGMc1(3:4,3:4);
        Mat(:,:,m*(i-1)+j)=diag(sqrt(are))*Gamma*(MGM+diag(kappa(m*(i-1)+j,:)./are'))*Gamma*diag(sqrt(are));
        variance_th(:,:,m*(i-1)+j)=ytilde0'*Mat(:,:,m*(i-1)+j)*ytilde0;
    end
end
for ze=1
score_int=[];ze_w=1:m;ze_w(ze)=[];Sigma_int=[];
for i=ze_w
    score_int=[score_int;score_th(m+ze,ze)-score_th(m+ze,i)];Sigma_int2=[];
    for j=ze_w
        Big=variance_th([ze,i],[ze,j],m+ze);
        Sigma_int2=[Sigma_int2 Big(1,1)+Big(2,2)-Big(1,2)-Big(2,1)];
    end
    Sigma_int=[Sigma_int;Sigma_int2];
end
%Sigma_int=er;score_int=[mean(score1(:,1)-score1(:,2));mean(score1(:,1)-score1(:,3))];
error(ze)=1-mvncdf(zeros(m-1,1),inf*ones(m-1,1),score_int,Sigma_int);
end
evec=[];
for kc=2:m
    ei=zeros(m,1);ei(1)=1;ej=zeros(m,1);ej(kc)=1;
    evec=[evec;ei'-ej'];
end
e21=zeros(k*m,1);e21(m+1)=1;
score_int2=evec*ytilde0'*(eye(m*k)-diag(deltabar)^(1/2)*Gamma*diag(deltabar)^(-1/2))*e21;
Sigma_int2=evec*ytilde0'*Mat(:,:,m+1)*ytilde0*evec';
evec*ytilde0'*Mat(:,:,m+1)*ytilde0*evec'
%weight=[0.3 0.5 0.2];
weight=ones(m,1)./m;
obj=@(ystar) 0;
for i=1:m
    i
    evec=[];i_w=1:m;i_w(i)=[];
    for kc=i_w
        ei=zeros(m,1);ei(i)=1;ej=zeros(m,1);ej(kc)=1;
        evec=[evec;ei'-ej'];
    end
    e21=zeros(k*m,1);e21(m+i)=1;
    proba_the(i)=mvncdf(zeros(m-1,1),inf*ones(m-1,1),evec*ytilde0'*(eye(m*k)-diag(deltabar)^(1/2)*Gamma*diag(deltabar)^(-1/2))*e21,evec*ytilde0'*Mat(:,:,m+i)*ytilde0*evec');
    obj=@(ystar) obj(ystar)-weight(i)*mvncdf(zeros(m-1,1),inf*ones(m-1,1),evec*ystar'*(eye(m*k)-diag(deltabar)^(1/2)*Gamma*diag(deltabar)^(-1/2))*e21,evec*ystar'*Mat(:,:,m+i)*ystar*evec');
end
x0=ytilde0;
 [y_opti,~]=fminsearch(obj,x0);
 Ze=[eye(m)-ones(m,1)*nt(1:m)'/n1 zeros(m,m);zeros(m,m) eye(m)-ones(m,1)*nt(m+1:m*k)'/n2];
y_opt=pinv(Ze)*y_opti;
  score_int3=evec*y_opti'*(eye(m*k)-diag(deltabar)^(1/2)*Gamma*diag(deltabar)^(-1/2))*e21;
 Sigma_int3=evec*y_opti'*Mat(:,:,m+1)*y_opti*evec';

 mvncdf(zeros(m-1,1),inf*ones(m-1,1),score_int3,Sigma_int3) 
 mvncdf(zeros(m-1,1),inf*ones(m-1,1),score_int2,Sigma_int2) 
 [~,pred]=max(score1,[],2);

 for i=1:m
     error_emp(i)=sum(pred(1+10000*(i-1):10000*i)~=i*ones(10000,1))./10000;
 end
%   1-error_emp
%   err_the
%   obj0=obj(ytilde0)
%   objst=obj(y_opti)
% 
%    error_emp=1-sum(error)/m;
   
end
