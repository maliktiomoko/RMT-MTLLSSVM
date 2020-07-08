function [X1,X2] = scaled_data(X1,X2,cas)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
switch cas
    case '1'
        p=size(X1,1);
        X1=(X1)./((1/p)*sqrt(trace((X1-mean(X1))'*(X1-mean(X1)))));
        X2=(X2)./((1/p)*sqrt(trace((X2-mean(X2))'*(X2-mean(X2)))));
    case '2'
        X1=(X1)./sqrt(norm(X1));
        X2=(X2)./sqrt(norm(X2));
    case '3'
        X1=(X1)./repmat(sqrt(sum(X1.^2,2)),1,size(X1,2));
        X2=(X2)./repmat(sqrt(sum(X2.^2,2)),1,size(X2,2));
    case '4'
        p=size(X1,1);
        n1=size(X1,2);n2=size(X2,2);
        P1=(eye(n1)-(1/n1)*ones(n1,1)*ones(1,n1));
        P2=(eye(n2)-(1/n2)*ones(n2,1)*ones(1,n2));
        X1=(X1*P1)./((1/(p*n1))*(trace(X1*P1*P1'*X1')));
        X2=(X2*P2)./((1/(p*n2))*(trace(X2*P2*P2'*X2')));
end

