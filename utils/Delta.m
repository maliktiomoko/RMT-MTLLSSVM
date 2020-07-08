function [out] = Delta(A,c,k,p,Ct)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
epsi=1e-3;
out1=rand();out2=rand();n_iter=0;
while (sum(abs(out2-out1)>epsi*ones(1,1))>0 && n_iter<500)
    n_iter
    out1=out2;
    invQtilde=(k/(k*c))*(A^(1/2)*kron(eye(k),Ct)*A^(1/2))/((1+out1));
    invQtildez=(invQtilde+eye(k*p));
    d=zeros(k,1);d(1)=1;d=d*d';
    out2=(1/(k*p))*trace((A^(1/2)*(kron(d,Ct))*A^(1/2))/invQtildez);
    n_iter=n_iter+1;
end
out=out2;

end

