function [risk_train,risk_train_th,risk_test,risk_test_th] = MLSSVRTrain_th1_multi_output_regression(trnX,trnY, gamma, lambda,Ct,tstX1,y_test,k,n_test,beta,a,sigma,epsilon)
    % Function that computes theoretically the means and the variance of the score for MTL
    %Input:
    %Output: theoretical error/Empirical error/alpha/b/Theoretical
    %mean/Theoretical variance/Empirical mean/ Empirical variance
    n=size(trnX,2);p=size(trnX,1); c=(k*p)/(n);
%     k=size(trnY,2);
    Z=[];
    for task=1:k
        Z=blkdiag(Z,trnX);
    end
    P = zeros(k*n, k); 
    P(1:n,1)=ones(n,1);
    for j=1:k-1
        P(1+j*n:(j+1)*n,j+1)=ones(n,1);
    end
    tstN1 = size(tstX1, 1);
    P_t = zeros(k*tstN1, k); 
    P_t(1:tstN1,1)=ones(tstN1,1);
    for j=1:k-1
        P_t(1+j*tstN1:(j+1)*tstN1,j+1)=ones(tstN1,1);
    end
    A=kron((diag(gamma)+sqrt(lambda)*sqrt(lambda)'),eye(p));
%          lambda=lambda./norm(A);gamma=gamma./norm(A);
%      %A=kron((diag(gamma)+lambda*ones(k,1)*ones(1,k)),eye(p));
%      A=kron((diag(gamma)+sqrt(lambda)*sqrt(lambda)'),eye(p));
for i=1:k
    for j=1:2
        d=zeros(k,1);d(i)=1;d=d*d';
         C(:,:,2*(i-1)+j)=A^(1/2)*(kron(d,Ct))*A^(1/2);
%         C(:,:,m*(i-1)+j)=(k*p)*A^(1/2)*(kron(d,X11{j,i}'*X11{j,i}/nt(m*(i-1)+j)))*A^(1/2);
    end
end
     H=(Z'*A*Z+eye(k*n));
    Dtilde=(A^(1/2)*(Z*Z')*A^(1/2)+eye(k*p))^(-1);
    eta = H \ P; 
    nu = H \ trnY(:); 

    S = P'*eta;
    b = (S\eta')*trnY(:);
    b_th=P'*trnY(:)/n;
    beta1=[];
    for i=1:k
        beta1=[beta1;beta{i}'];
    end
    Pc=kron(eye(k),(eye(n)-ones(n,1)*ones(1,n)/n));
    yPb=Pc*(Z'*beta1+0*rand(k*n,1));
    
    ytilde0=trnY(:)-P*b;
    alpha2 = nu - eta*b;
    score_training=(eye(k*n)-inv(H))*(trnY(:)-P*b)+P*b;
         out1=Delta(A,c,k,p,Ct);
     invQtilde=(k/(k*c))*(A^(1/2)*kron(eye(k),Ct)*A^(1/2))/((1+out1));
     Qtildez=inv(invQtilde+eye(k*p));
      %[out,Q] = delta_F(p,1,C,(1/k)*ones(2*k,1),c,2,'synthetic');
          Cto=Ct;ei=zeros(k,1);ei(1)=1;
     C=A^(1/2)*kron(eye(k),Cto)*A^(1/2);
     Tb=trace(C*Qtildez^2)/(k*p);
     Cg=trace(C*Qtildez*C*Qtildez)/(k*p);
     dt=n/(k*p*(1+out1)^2);
     T=Tb./(1-dt*Cg);
     det=Qtildez^2+dt*T*Qtildez*C*Qtildez;
     det_ran=Dtilde^2;
     det_ran2=inv(H)^2;
     detQ=(1./(1+out1))*eye(k*n);
     C_tota=(n/(k*p))*A^(1/2)*kron(eye(k),Cto)*A^(1/2);
     
     test1=Z'*A^(1/2)*Dtilde^2*A^(1/2)*Z;
     test2=(inv(H)-test1);
     detQ2=detQ-eye(k*n)*(trace(C_tota*det))/(k*n*(1+out1)^2);
     %a=randn(k*n,1);
     Mat_var=kron(diag(sigma),eye(n));
     risk_train=sum((trnY(:)-score_training).^2)/(k*n);
    risk_train_th_1=((trnY(:)-P*b)'*(inv(H)*inv(H))*(trnY(:)-P*b))/(k*n);
    risk_train_th_1b=(trnY(:)'*Pc*(inv(H)*inv(H))*Pc*trnY(:))/(k*n);
    risk_train_th_1bb=((beta1'*Z+a')*Pc*(inv(H)*inv(H))*Pc*(Z'*beta1+a))/(k*n);
      risk_train_th_2=(beta1'*Z*Pc*(inv(H)*inv(H))*Pc*Z'*beta1)/(k*n)+...
          (beta1'*Z*Pc*(inv(H)*inv(H))*Pc*a)/(k*n)+...
          (a'*Pc*(inv(H)*inv(H))*Pc*(Z'*beta1))/(k*n)+...
          (a'*Pc*(inv(H)*inv(H))*Pc*a)/(k*n);
      risk_train_th_22=(beta1'*Z*Pc*(inv(H)*inv(H))*Pc*Z'*beta1)/(k*n)+...
          (a'*Pc*(inv(H)*inv(H))*Pc*a)/(k*n);
      risk_train_th_3=(beta1'*A^(-1/2)*(Dtilde)*A^(-1/2)*beta1)/(k*n)-...
          (beta1'*A^(-1/2)*(det_ran)*A^(-1/2)*beta1)/(k*n)+...
          1*trace(Pc*det_ran2)/(k*n);
      risk_train_th=(beta1'*A^(-1/2)*(Qtildez)*A^(-1/2)*beta1)/(k*n)-...
          (beta1'*A^(-1/2)*(det)*A^(-1/2)*beta1)/(k*n)+...
          1*trace(Mat_var*Pc*detQ2)/(k*n);
%     Ran=(inv(H)*inv(H));
%     Ran_t=kron((diag(gamma)+sqrt(lambda)'*sqrt(lambda))^2,inv(trnX'*trnX+eye(n))^2);
%      m=@(z) (1-c-z)./(2*c*z)-(1/(2*c*z)).*sqrt((1-c-z).^2-4*c*z);
%     mprim=@(z) m(z).^2.*(1+c*m(z))./(1-c*z.*m(z).^2);
%     Ran_det=mprim(-1)*kron((diag(gamma)+sqrt(lambda)'*sqrt(lambda))^2,eye(n));
%     risk_train_th=(trnY(:)-P*b)'*Ran_det*(trnY(:)-P*b)/(k*n);
%     ei=zeros(k,1);ei(1)=1;
%     Cto=eye(p);
%     C=kron(ei*ei',Cto);
%     Tb=trace(C*Qtildez^2)/(k*p);
%     Cg=trace(C*Qtildez*C*Qtildez)/(k*p);
%     dt=n/(k*p*(1+out1)^2);
%     T=Tb./(1-dt*Cg);
%     val=(T/(1+out1)^2)*trace((trnY(:)-P*b)*(trnY(:)-P*b)');
%     ei=zeros(k,1);ei(k)=1;
%     eps=rand(k*tstN1,1);
     test1=kron(eye(k),tstX1);
      risk_test=sum((y_test(:)-(test1*A*Z*alpha2)-P_t*b).^2)/(k*tstN1);
        %risk_test_th_1=sum((y_test(:)-P_t*b-(test1*A^(1/2)*Dtilde*A^(1/2)*Z*(trnY(:)-P*b))).^2)/(k*tstN1);
         %risk_test_th_2=sum((y_test(:)-P_t*b-(test1*A^(1/2)*Dtilde*A^(1/2)*Z*(trnY(:)))).^2)/(k*tstN1);
        risk_test_th_1=sum((test1*beta1+epsilon-P_t*b-(test1*A^(1/2)*Dtilde*A^(1/2)*Z*(Z'*beta1+a))).^2)/(k*tstN1);
        risk_test_th_11=sum((test1*beta1-(test1*A^(1/2)*Dtilde*A^(1/2)*Z*(Z'*beta1+a))).^2)/(k*tstN1)+...
            sum((epsilon-P_t*b).^2)/(k*tstN1);
        risk_test_th_12=sum((test1*beta1-(test1*A^(1/2)*Dtilde*A^(1/2)*Z*(Z'*beta1+a))).^2)/(k*tstN1)+...
            (sum(sigma)/k)+b'*b/(k);
        risk_test_th_2=(1/(k*p))*((beta1-(A^(1/2)*Dtilde*A^(1/2)*Z*(Z'*beta1+a)))'*(kron(eye(k),Cto))*(beta1-(A^(1/2)*Dtilde*A^(1/2)*Z*(Z'*beta1+a))))/k+...
            (sum(sigma)/k)+b'*b/(k);
       risk_test_th_3=(1/(k*p))*beta1'*A^(-1/2)*Dtilde*A^(1/2)*kron(eye(k),Cto)*A^(1/2)*Dtilde*A^(-1/2)*beta1/k+...
           (1/(k*p))*trace(a*a'*Z'*A^(1/2)*Dtilde*A^(1/2)*(kron(eye(k),Cto))*(A^(1/2)*Dtilde*A^(1/2)*Z))/k+(sum(sigma)/k)+b'*b/(k);
        risk_test_th_5=(1/(k*p))*beta1'*A^(-1/2)*Dtilde*A^(1/2)*kron(eye(k),Cto)*A^(1/2)*Dtilde*A^(-1/2)*beta1/k+...
           (1/(k*p))*trace(Z'*A^(1/2)*Dtilde*A^(1/2)*(kron(eye(k),Cto))*(A^(1/2)*Dtilde*A^(1/2)*Z))/k+...
           (sum(sigma)/k)+b'*b/(k);
   DAD=Dtilde*A^(1/2)*kron(eye(k),Cto)*A^(1/2)*Dtilde;
   Ran2=(Z'*A^(1/2)*Dtilde*A^(1/2)*(kron(eye(k),Cto))*(A^(1/2)*Dtilde*A^(1/2)*Z));
   Tb2=trace(C*Qtildez*A^(1/2)*kron(eye(k),Cto)*A^(1/2)*Qtildez)/(k*p);
     dt=n/(k*p*(1+out1)^2);
     T2=Tb2./(1-dt*Cg);
   DAD_eq=Qtildez*A^(1/2)*kron(eye(k),Cto)*A^(1/2)*Qtildez+dt*T2*Qtildez*C*Qtildez;
    %Ran_eq=eye(k*n)*(trace(C_tota*DAD_eq))/((1+out1)^2);
    Ran_eq=eye(k*n)*(trace(C_tota*DAD_eq))/(k*n*(1+out1)^2);
    %Ran_eq=eye(k*n)*(trace(A^(1/2)*Z*Z'*A^(1/2)*DAD))/(n*(1+out1)^2);
    risk_test_ran=(1/(k*p))*beta1'*A^(-1/2)*DAD*A^(-1/2)*beta1/k+...
          (1/(k*p))*trace(Ran2)/k+(sum(sigma)/k)+b'*b/(k);
    risk_test_th=(1/(k*p))*beta1'*A^(-1/2)*DAD_eq*A^(-1/2)*beta1/k+...
          (1/(k*p))*trace(Ran_eq)/k+(sum(sigma)/k)+b'*b/(k);
%       risk_test_th=beta1'*beta1+beta1'*A^(-1/2)*Dtilde*A^(1/2)*kron(eye(k),Cto)*A^(1/2)*Dtilde*A^(-1/2)*beta1+a'*Z'*A^(-1/2)*Dtilde*A^(1/2)*kron(eye(k),Cto)*A^(1/2)*Dtilde*A^(1/2)*Z*a
% %      risk_testp=trace((y_test(:)-score_test)'*(y_test(:)-score_test))/(k*tstN1);
%       risk_test_th=trace((y_test(:))'*(y_test(:)))/(k*tstN1)+trace(score_test(:)'*score_test(:))/(k*tstN1)-...
%           2*trace(y_test(:)'*score_test(:))/(k*tstN1);
%       risk_test_th=trace((y_test(:))'*(y_test(:)))/(k*tstN1)+trace((score_test(:))'*(score_test(:)))/(k*tstN1)-...
%           2*trace(y_test(:)'*score_test)/(k*tstN1);
%       hg=trace(C*Qtildez*A*Qtildez)/(k*p);
%         h=hg+dt*T*Cg;
%       risk_test_th=trace((y_test(:))'*(y_test(:)))/(k*tstN1)+(h/((1+out1)^2))*trace((trnY(:)-P*b)'*(trnY(:)-P*b))/(k^2*p)-...
%           2*trace(y_test(:)'*score_test)/(k*tstN1)++...
%           trace(P_t*b*b'*P_t')/(k*tstN1);
%      risk_test1=(1/sqrt(k*p))*(test1*A*Z*(H\(trnY(:)-P*b)))+P_t*b;
%      term1=trace((score_test(:))'*(score_test(:)))/(k*tstN1)
%      risk_test1=trace((test1*A*Z*(H\(trnY(:)-P*b))*((trnY(:)-P*b)'/H)*Z'*A*test1')/(k*p*tstN1*k))+...
%          trace(P_t*b*b'*P_t')/(k*tstN1)
        
        %risk_test_th=trace((y_test(:)-P_t*b)'*(y_test(:)-P_t*b))/(k*tstN1)+...
         %(h/((1+out1)^2))*trace((trnY(:)-P*b)'*(trnY(:)-P*b))/(k^2*p);
%      term2=trace((test1*A*Z*(H\(trnY(:)-P*b))*((trnY(:)-P*b)'/H)*Z'*A*test1')/(k*p*tstN1*k));
%      score4=trace(kron(eye(k),Cto)*A*Z*(H\(trnY(:)-P*b))*((trnY(:)-P*b)'/H)*Z'*A)/(k^2*p)
%      Eq=A*Z*(H\(trnY(:)-P*b))*((trnY(:)-P*b)'/H)*Z'*A/(k^2*p);
%      Eq_test=Dtilde*A^(1/2)*Z*(trnY(:)-P*b)*(trnY(:)-P*b)'*Z'*A^(1/2)*Dtilde/(k*p);
%      Eq_test2=(trace(C*Qtildez^2)/(k*p*(1+out1)^2))*trace((trnY(:)-P*b)'*(trnY(:)-P*b))/(k*p)
%      
%      Eq_test2=(h/((1+out1)^2))*trace((trnY(:)-P*b)'*(trnY(:)-P*b))/(k^2*p)
%      test=1
%   Ex=trace(A^(1/2)*Z*Z'*A^(1/2)*DAD_eq)/(k*n*(1+out1)^2)
     
end
