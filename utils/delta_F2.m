function [out] = delta_F(p,param,C,c,co,m,cov,data)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
switch cov
    case 'general'
        switch data
            case 'synthetic'
        epsi=1e-3;
        k=size(C,3)/m;
        out1=rand(m*k,1);out2=rand(m*k,1);n_iter=0;
        while (sum(abs(out2-out1)>epsi*ones(m*k,1))>0 && n_iter<500)
            %n_iter
            out1=out2;
            invQtilde=zeros(k*p,k*p);
            for i=1:k
                for j=1:m
                    invQtilde=invQtilde+(c(m*(i-1)+j)/co)*squeeze(C(:,:,m*(i-1)+j))/((1+out1(m*(i-1)+j)));
                end
            end
            invQtildez=(invQtilde+eye(k*p));
            for i=1:k
                for j=1:m
                    out2(m*(i-1)+j)=(1/(k*p))*trace(C(:,:,m*(i-1)+j)/invQtildez);
                end
            end
            n_iter=n_iter+1;
        end
        out=out2;
            case 'real'
                epsi=1e-10;
                k=size(C,3)/m;
                out1=rand(m*k,1);out2=rand(m*k,1);n_iter=0;
                while (sum(abs(out2-out1)>epsi*ones(m*k,1))>0 && n_iter<500)
                    %n_iter
                    %n_iter
                    out1=out2;
                    invQtilde=zeros(k*p,k*p);
                    for i=1:k
                        for j=1:m
                            invQtilde=invQtilde+(c(m*(i-1)+j)/co)*squeeze(C(:,:,m*(i-1)+j))/((1+out1(m*(i-1)+j)));
                        end
                    end
                    invQtildez=((k*p)*invQtilde+eye(k*p));
                    for i=1:k
                        for j=1:m
                            out2(m*(i-1)+j)=trace(C(:,:,m*(i-1)+j)/invQtildez);
                        end
                    end
                    n_iter=n_iter+1;
                end
                out=out2;
        end
case 'identity'
        switch data
            case 'synthetic'
                epsi=1e-10;
            k=size(C,3)/m;
            out1=rand(k,1);out2=rand(k,1);n_iter=0;
            while (sum(abs(out2-out1)>epsi*ones(k,1))>0 && n_iter<500)
                %n_iter
                out1=out2;
                Agotique=(1/k)*inv(diag(param.nt./(k*p*(1+out1)))+inv(diag(param.gamma)+param.lambda*ones(k,1)*ones(1,k)));
                for i=1:k
                    out2(i)=Agotique(i,i);
                end
                n_iter=n_iter+1;
            end
            out=out2;
%             Qtildez=[];
            case 'real'
                    epsi=1e-10;
                    k=size(C,3)/m;
                    out1=rand(k,1);out2=rand(k,1);n_iter=0;
                    while (sum(abs(out2-out1)>epsi*ones(k,1))>0 && n_iter<500)
                        %n_iter
                        out1=out2;
                        Agotique=p*inv(diag(param.nt./((1+out1)))+inv(diag(param.gamma)+param.lambda*ones(k,1)*ones(1,k)));
                        for i=1:k
                            out2(i)=Agotique(i,i);
                        end
                        n_iter=n_iter+1;
                    end
                    out=out2;
        end
end
end