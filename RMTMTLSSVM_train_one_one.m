function [acc,acc_opt,acc_th,acc_th_opt] = RMTMTLSSVM_train_one_one(Xs,ys,Xt,yt,X_test,y_true,m,M,Ct,nst)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
k=2;M11=[];M22=[];X1{1}=[];X2=[];
Xttot=[Xt X_test];
% [Xs,Xttot]=scaled_data(Xs,Xttot,'1');
Xt=Xttot(:,1:size(Xt,2));X_test=Xttot(:,size(Xt,2)+1:end);
for task=1:k-1
    for i=1:m
        X11{i,task}=Xs(:,ys==i)';
        X1{task}=[X1{task} X11{i,task}'];
        M11=[M11 mean(X11{i,task})'];
        ns(m*(task-1)+i)=size(X11{i,task},1);
    end
end
for i=1:m
    X22{i}=Xt(:,yt==i)';
    X2=[X2 X22{i}'];
	ns(i+m*(k-1))=floor(size(X22{i},1));
    M22=[M22 mean(X22{i})'];
end
%  M=[M11 M22];
X=[X1{1} X2];
p=size(X,1);
X=X/sqrt(k*p);
X1{1}=X1{1}/sqrt(k*p);X2=X2/sqrt(k*p);
gamma=[0.1;1];lambda=1;
% score_th=zeros(m,2*k);score_th_opt=zeros(m,2*k);
% variance_th=zeros(m,2*k);variance_th_opt=zeros(m,2*k);
% %error_s=zeros(m,1);error_t=zeros(m,1);error_s_opt=zeros(m,1);error_t_opt=zeros(m,1);
% y_opt=zeros(2*k,m);y_opt_opt=zeros(2*k,m);
% init_order=1:k*m;
% %score_test=zeros(Nb*m,m);
% lambda_opt=zeros(m,1);
% %
% n=sum(ns);gamma=[0.1;1];lambda=1;
% nsa=zeros(1,2*k);
% nsi=reshape([zeros(1,k);reshape(ns,m,k)],1,k*m+k);
Matrix=combntns(1:m,2);score_opt=[];score=[];
for i=1:size(Matrix,1)
    i
    c1=Matrix(i,1);c2=Matrix(i,2);
%     nsn=ns(init_order);
    M1s=M(:,c1); Cts(:,:,1)=Ct(:,:,c1);
    M1t=M(:,m+c1);Cts(:,:,3)=Ct(:,:,m+c1);
    M2s=M(:,c2);Cts(:,:,2)=Ct(:,:,c2);
    M2t=M(:,m+c2);Cts(:,:,4)=Ct(:,:,c2+m);
    Ms=[M1s M2s M1t M2t];
    ns2=[0 ns];
    %nsa=[ns(i) sum(ns(1:m))-ns(i) ns(i+m) sum(ns(m+1:k*m))-ns(i+m)];
    rg_1=sum(ns2(1:m+c1))+1:sum(ns2(1:m+c1+1));rg_1=rg_1(randperm(length(rg_1)));
    rg_2=sum(ns2(1:m+c2))+1:sum(ns2(1:m+c2+1));rg_2=rg_2(randperm(length(rg_2)));
    rg_3=sum(ns2(1:c1))+1:sum(ns2(1:c1+1));rg_3=rg_3(randperm(length(rg_3)));
    rg_4=sum(ns2(1:c2))+1:sum(ns2(1:c2+1));rg_4=rg_4(randperm(length(rg_4)));
    Xt2=[X(:,rg_1) X(:,rg_2)];
    Xt1{1}=[X(:,rg_3) X(:,rg_4)];
    nn=[ns(c1);ns(c2);ns(m+c1);ns(m+c2)];
    J=zeros(sum(nn),2*k);
    for h=1:2*k
        J(sum(nn(1:h-1))+1:sum(nn(1:h)),h)=ones(nn(h),1);
    end
    %tildey=-ones(2*k,1);tildey(1:2:end)=1;
    tildey=[1;-1;1;-1];
    yc=J*tildey;
%     figure
%     plot(X1(1,:),X1(2,:),'r*');hold on;plot(X2(1,:),X2(2,:),'g*')
Jk=zeros(sum(nn),2*k);
for g=1:2*k
    Jk(sum(nn(1:g-1))+1:sum(nn(1:g)),g)=ones(nn(g),1);
end
%        [score1(:,i),pred(:,i),error_th(:,i),alpha2, b,score_th(i,:),variance_th(i,:),score_emp,var_emp,y_opt(:,i)] = MLSSVRTrain_th1_centered(X1,X2, yc, gamma, lambda,Ms,X_test,nn,'task');
    N_te=[0;nst];
    X_test2=[];
    for gh=1:m
        rg_test=1+sum(N_te(1:gh)):sum(N_te(1:gh+1));
        rg_test=rg_test(randperm(length(rg_test)));
        X_test2=[X_test2 X_test(:,rg_test)];
    end
       [score1,error_opt,error_th(i,:),error_emp,alpha2, b,score_th,variance_th1,score_emp,var_emp,y_opt(:,i),pred(:,i),obj1] = MLSSVRTrain_th1_centered_other(Xt1,Xt2,yc, gamma, lambda,Ms,Cts,X_test2,nn,'task',k,nst(1:2));
        [err(i,:),scf,ver] = MLSSVRTrain_th1_centered_other_class_one_vs_one(Xt1,Xt2,yc, gamma, lambda,Ms,Cts,X_test2,nn,'task',k,nst,M,m,Jk,Ct,ns);
%        error=@(x) perf_multi(X1,X2, [x(1);x(2)], x(3),Ms,nn);
%        init0=[0.1;1;1];
%        [param_opt,error_optim]=fmincon(error,init0,[-1 0 0;0 -1 0;0 0 -1],[0;0;0]);
%        gamma_opt1(i)=param_opt(1);gamma_opt2(i)=param_opt(2);lambda_opt(i)=param_opt(3);
       yopt=Jk*y_opt(:,i);
%        [score1_opt(:,i),pred_opt(:,i),error_th_opt(:,i),alpha2_opt, b_opt,score_th_opt(i,:),variance_th_opt(i,:),score_emp_opt,var_emp_opt,y_opt_opt(:,i)] = MLSSVRTrain_th1_centered(X1,X2, yopt, [gamma(1);gamma(2)], lambda,Ms,X_test,nn,'task');
       [score1_opt,error_opt_opt,error_th_opt(i,:),error_emp_opt,alpha2, b,score_th_opt,variance_th_opt,score_emp_opt,var_emp_opt,y_opt_opt,pred_opt(:,i),obj1] = MLSSVRTrain_th1_centered_other(Xt1,Xt2,yopt, gamma, lambda,Ms,Cts,X_test2,nn,'task',k,nst(1:2));
        [err_opt(i,:),scf,ver] = MLSSVRTrain_th1_centered_other_class_one_vs_one(Xt1,Xt2,yopt, gamma, lambda,Ms,Cts,X_test2,nn,'task',k,nst,M,m,Jk,Ct,ns);
       scorea=pred_opt(:,i);scorey=pred_opt(:,i);scorea(scorey<0)=c2;scorea(scorey>0)=c1;
       score_opt=[score_opt scorea];
       scoreb=pred(:,i);scoreby=pred(:,i);scoreb(scoreby<0)=c2;scoreb(scoreby>0)=c1;
       score=[score scoreb];
end
error_theo=error_th(:,3:4);error_theo_opt=error_th_opt(:,3:4);
for i=1:m
    error_trom(i)=prod(1-error_theo(Matrix==i));
end
 pred=mode(score,2);
 for i=1:size(Matrix,1)
     input{i}=Matrix(i,:);
 end
 Out=allcomb(input{:});
 for j=1:m
     Out_res=unique(Out(find(mode(Out,2)==j),:),'rows');
     %Out_res=unique(Out_restricted(:,any(Out_restricted==j)),'rows');
     %OUT=unique(Out(:,any(Out_restricted==j)),'rows');
     %Matrix_res=Matrix(any(Out_restricted==j),:);
     Matrix_res=Matrix;
     OUT=Out;
      Proba=zeros(size(Out_res));Proba_tot=zeros(size(OUT));
      Proba_out=zeros(size(Out_res));Proba_tot_out=zeros(size(OUT));
     for h=1:size(Out_res,2)
         if any(Matrix_res(h,1)==j)
         if Matrix_res(h,1)==j
            Proba(Out_res(:,h)==Matrix_res(h,1),h)=1-error_theo(h,1);
            Proba_out(Out_res(:,h)==Matrix_res(h,1),h)=1-error_theo_opt(h,1);
         else
             Proba(Out_res(:,h)==Matrix_res(h,1),h)=error_theo(h,1);
             Proba_out(Out_res(:,h)==Matrix_res(h,1),h)=error_theo_opt(h,1);
         end
         if Matrix_res(h,2)==j
            Proba(Out_res(:,h)==Matrix_res(h,2),h)=1-error_theo(h,2);
            Proba_out(Out_res(:,h)==Matrix_res(h,2),h)=1-error_theo_opt(h,2);
         else
             Proba(Out_res(:,h)==Matrix_res(h,2),h)=error_theo(h,2);
             Proba_out(Out_res(:,h)==Matrix_res(h,2),h)=error_theo_opt(h,2);
         end
         else
             Proba(Out_res(:,h)==Matrix_res(h,1),h)=err(h,j);
             Proba(Out_res(:,h)==Matrix_res(h,2),h)=1-err(h,j);
             Proba_out(Out_res(:,h)==Matrix_res(h,1),h)=err_opt(h,j);
             Proba_out(Out_res(:,h)==Matrix_res(h,2),h)=1-err_opt(h,j);
         end
%          if Matrix_res(h,2)==j
%             Proba_out(Out_res(:,h)==Matrix_res(h,2),h)=1-error_theo_opt(h,2);
%          else
%              Proba_out(Out_res(:,h)==Matrix_res(h,2),h)=error_theo_opt(h,2);
%          end
             
     end
%       for h=1:size(OUT,2)
%           if Matrix_res(h,1)==j
%             Proba_tot(OUT(:,h)==Matrix_res(h,1),h)=1-error_theo(h,1);
%          else
%              Proba_tot(OUT(:,h)==Matrix_res(h,1),h)=error_theo(h,1);
%          end
%          if Matrix_res(h,2)==j
%             Proba_tot(OUT(:,h)==Matrix_res(h,2),h)=1-error_theo(h,2);
%          else
%              Proba_tot(OUT(:,h)==Matrix_res(h,2),h)=error_theo(h,2);
%          end
%          if Matrix_res(h,1)==j   
%             Proba_tot_out(OUT(:,h)==Matrix_res(h,1),h)=1-error_theo_opt(h,1);
%          else
%              Proba_tot_out(OUT(:,h)==Matrix_res(h,1),h)=error_theo_opt(h,1);
%          end
%          if Matrix_res(h,2)==j
%             Proba_tot_out(OUT(:,h)==Matrix_res(h,2),h)=1-error_theo_opt(h,2);
%          else
%              Proba_tot_out(OUT(:,h)==Matrix_res(h,2),h)=error_theo_opt(h,2);
%          end
%       end
     proba(j)=sum(prod(Proba,2));proba_opt(j)=sum(prod(Proba_out,2));
 end
 Nb=nst(1);
 error1(1)=sum(pred(1:Nb)~=y_true(1:Nb)')./Nb;
for i=1:m-1
    error1(i+1)=sum(pred(1+Nb*i:Nb*(i+1))~=y_true(1+Nb*i:Nb*(i+1))')./Nb;
end
error1
1-proba
error=sum(pred~=y_true');
pred_opt=mode(score_opt,2);
error1_opt(1)=sum(pred_opt(1:Nb)~=y_true(1:Nb)')./Nb;
for i=1:m-1
    error1_opt(i+1)=sum(pred_opt(1+Nb*i:Nb*(i+1))~=y_true(1+Nb*i:Nb*(i+1))')./Nb;
end
error1_opt
1-proba_opt
error_opti=sum(pred_opt~=y_true');
acc=1 - error/size(X_test,2);
acc_opt=1 - error_opti/size(X_test,2);
acc_th=nst'*proba'./sum(nst);
acc_th_opt=nst'*proba_opt'./sum(nst);

end