function [score1,error_emp,alpha2, b] = Least_square_svm(trnX,trnY, gamma,M,tstX1,nt,n_test)
% Function that computes theoretically the means and the variance of the score for MTL
%Input:
%Output: theoretical error/Empirical error/alpha/b/Theoretical
%mean/Theoretical variance/Empirical mean/ Empirical variance
    [p,n]=size(trnX);
    trnY(trnY==2)=-1;
    P1=[];
    P1=blkdiag(P1,eye(n)-(1/n)*ones(n,1)*ones(1,n));
    Z=trnX*P1;
    Moy_gen=zeros(size(M,1),1);
    for fr=1:2
        Moy_gen=Moy_gen+(nt(fr)/n)*M(:,fr);
    end
    tstX1=tstX1-Moy_gen;
    H=(Z'*Z+(1/gamma)*eye(n));
    eta = H \ ones(n,1); 
    nu = H \ trnY(:); 
 
    S = ones(n,1)'*eta;
    b = (S\eta')*trnY(:);
    alpha2 = nu - eta*b;
    tstN1 = size(tstX1, 2);
    score1=(1/sqrt(p))*(tstX1'*Z*alpha2)+ones(tstN1,1)*b;
  figure
    hold on
  histogram(real(score1(1:n_test(1),1)),80,'Normalization','probability')
  histogram(real(score1(n_test(1)+1:sum(n_test),1)),80,'Normalization','probability')
 pred1=zeros(size(tstX1,2),1);
 moy=(mean(score1(1:n_test(1)))+mean(score1(n_test(1)+1:sum(n_test))))/2;
 pred1(score1>moy)=1;pred1(score1<moy)=-1;
 error_emp(1)=sum(pred1(1:n_test(1))~=ones(n_test(1),1))/n_test(1);
 error_emp(2)=sum(pred1(n_test(1)+1:sum(n_test))~=-ones(n_test(2),1))/n_test(2);
end

