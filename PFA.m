%%%%%%%%%%%%%%%%% Model ML SVR %%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all
clc
data='real';
addpath('./utils');
addpath('./datasets');
n_training_sample=500;
switch data
    case 'synthetic'
         p=128;
         %ns=floor([1.3 1.7 2.3 2.2]*p);
         ns=[384,256,64,40];
         nst=10000*ones(4,1);
         X1=[];
         mus1=[ 1; 1;0;0;zeros(p-4,1)];
         mus2=[-1;-1;0;0;zeros(p-4,1)];
         orth=[ 0; 0;1;1;zeros(p-4,1)];
         delta=0.5;
         mut1=[sqrt(1-delta^2); delta;0;0;zeros(p-4,1)]*sqrt(p);
         mut2=-mut1;
         M=[mus1,mus2];
         k=2;
        param=[0.0 0.0 0.0 0.0];
        for i=1:k
            for j=1:2
                C(:,:,2*(i-1)+j)=10*toeplitz(param(2*(i-1)+j).^(0:p-1));
            end
        end
        for i=1:k-1
            M=[M mut1 mut2];
        end
        for j=1:2
            X1 = [X1 M(:,j)+C(:,:,j)^(1/2)*randn(p,ns(j))];
        end
        X_test1 =  M(:,1)+C(:,:,1)^(1/2)*randn(p,nst(1));
        X_test2 =  M(:,2)+C(:,:,2)^(1/2)*randn(p,nst(2));
        X2=[];
        for j=1:2
            X2 = [X2 M(:,2*(k-1)+j)+C(:,:,2*(k-1)+j)^(1/2)*randn(p,ns(2*(k-1)+j))];
        end
        X_test3 = M(:,3)+C(:,:,3)^(1/2)*randn(p,nst(3));
        X_test4 =  M(:,4)+C(:,:,4)^(1/2)*randn(p,nst(4));
        n=sum(ns);
        m=2;
        J=zeros(n,m*k);
        for i=1:m*k
            J(sum(ns(1:i-1))+1:sum(ns(1:i)),i)=ones(ns(i),1);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%% TEST

        n_test=sum(nst);
        J_test=zeros(n_test,m*k);
        for i=1:m*k
            J_test(sum(nst(1:i-1))+1:sum(nst(1:i)),i)=ones(nst(i),1);
        end
        yc(1:ns(1))=1;yc(ns(1)+1:ns(1)+ns(2))=-1;
        yc(ns(1)+ns(2)+1:ns(1)+ns(2)+ns(3))=1;yc(ns(1)+ns(2)+ns(3)+1:sum(ns))=-1;
        %J_test=ones(n_test,1);
        %%% Simu %%%

        param=abs(rand(m*k,1));
        n1=ns(1)+ns(2);n2=ns(3)+ns(4);
        P1 = eye(n1);
        P2 = eye(n2);
        lambda=10;gamma1=0.1;gamma2=1;
       test1 = [X_test1];
        testm1 = [X_test2];
        test2 = [X_test3];
        testm2 = [X_test4];
        nst(1)=size(X_test1,2);nst(1)=size(X_test2,2);nst(1)=size(X_test3,2);nst(1)=size(X_test4,2);
        k=2;
        init=[0.1,1,1];
        X1=X1/sqrt(2*p);X2=X2/sqrt(2*p);
        wanted=logspace(-6,-2,10);
    case 'real'
        load('x_train.mat');
        load('x_test.mat');
        load('y_train.mat');
        load('y_test.mat');
        x_train(y_train==0,:)=[];y_train(y_train==0)=[];
        x_train = x_train';
        x_test = x_test';
        x_train=[x_train(:,1:n_training_sample) x_train(:,2223+1:2223+n_training_sample) x_train(:,2223+5788+1:2223+5788+n_training_sample) x_train(:,2223+5788+641+1:2223+5788+641+n_training_sample)];
        y_train=[y_train(1:n_training_sample) y_train(2223+1:2223+n_training_sample) y_train(2223+5788+1:2223+5788+n_training_sample) y_train(2223+5788+641+1:2223+5788+641+n_training_sample)];
        %y_test=y_test(1:100);
            %p=p_vec(h);
        %jr=3000;
        k=2;m=2;
        %ns(1:m*k)=randi(jr,m*k,1);
        %ns=floor([2.7*p 1.3*p 2.3*p 1.7*p]);



        % nst=1000*ones(4,1);
        X1=[];X2=[];M11=[];M22=[];
        c1=[3 4 1 2];
        for task=1:k-1
            for i=1:m
        %        X11{i,task}=Xsr{task}(:,ysr{task}==c1(i))';
                X11{i,task}=x_train(:,y_train==c1(i))';
                X1=[X1 X11{i,task}'];
                ns(m*(task-1)+i)=size(X11{i,task},1);
                M11=[M11 mean(X11{i,task})'];
                C(:,:,m*(task-1)+i)=(X11{i,task}-mean(X11{i,task},1))'*(X11{i,task}-mean(X11{i,task},1))/ns(m*(task-1)+i);
            end
        end
        for i=1:m
            X22{i}=x_train(:,y_train==c1(i+2))';

            ns(i+m*(k-1))=size(X22{i},1);
            X2=[X2 X22{i}(1:ns(i+m*(k-1)),:)'];
            M22=[M22 mean(X22{i}(1:ns(i+m*(k-1)),:))'];
            C(:,:,i+m*(k-1))=(X22{i}-mean(X22{i},1))'*(X22{i}-mean(X22{i},1))/ns(i+m*(k-1));
        end
         M=[M11 M22];
         %ns=[500;500;500;500]';
         nst=100*ones(4,1);
        p=size(X11{1,1},2);
%         for j=1:2
%             X1 = [X1 M(:,j)+C(:,:,j)^(1/2)*randn(p,ns(j))];
%         end
         X_test1 =  M(:,1)+C(:,:,1)^(1/2)*randn(p,nst(1));
         X_test2 =  M(:,2)+C(:,:,2)^(1/2)*randn(p,nst(2));
%         X2=[];X_test=[];
%         for j=1:2
%             X2 = [X2 M(:,2*(k-1)+j)+C(:,:,2*(k-1)+j)^(1/2)*randn(p,ns(2*(k-1)+j))];
%         end
%         X_test3 = M(:,3)+C(:,:,3)^(1/2)*randn(p,nst(3));
%         X_test4 =  M(:,4)+C(:,:,4)^(1/2)*randn(p,nst(4));

        %M=[M11 M22];
        X_test3=x_test(:,y_test==1);
        X_test4=x_test(:,y_test==2);
        %M_test=M;

        n=sum(ns);
        nte = size(x_test, 2);
        nst(1)=size(X_test1,2);nst(2)=size(X_test2,2);nst(3)=size(X_test3,2);nst(4)=size(X_test4,2);

        %k=length(ns);

        J=zeros(n,m*k);
        for i=1:m*k
            J(sum(ns(1:i-1))+1:sum(ns(1:i)),i)=ones(ns(i),1);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%% TEST
        %n = size(x_test, 2);
        test1 = [X_test1];
        testm1 = [X_test2];
        test2 = [X_test3];
        testm2 = [X_test4];

        n_test=sum(nst);
        J_test=zeros(n_test,m*k);
        for i=1:m*k
            J_test(sum(nst(1:i-1))+1:sum(nst(1:i)),i)=ones(nst(i),1);
        end
        yc(1:ns(1))=1;yc(ns(1)+1:ns(1)+ns(2))=-1;
        yc(ns(1)+ns(2)+1:ns(1)+ns(2)+ns(3))=1;yc(ns(1)+ns(2)+ns(3)+1:sum(ns))=-1;
        param=zeros(m*k,1);
        n1=ns(1)+ns(2);n2=ns(3)+ns(4);
        lambda=1e2;gamma1=1000;gamma2=100;
        k=2;
        init=[100,1000,2];
        X1=X1/sqrt(2*p);X2=X2/sqrt(2*p);
        wanted=logspace(-4,-1,10);
end
for v=1:length(wanted)
    v
    [error_th,error_th_task,error_emp_task,alpha_task, b_task,score_mean_task,variance_task,score_emp_task,var_emp_task,y_opt] = MLSSVRTrain_th1_centered_fixed_pro(X1,X2, yc', gamma1,gamma2, lambda,M,C,J,test1,testm1,test2,testm2,ones(nst(1),1),-ones(nst(2),1),ones(nst(3),1),-ones(nst(4),1),ns,'task',wanted(v),'general');
     yop=[y_opt(1)*ones(1,ns(1)) y_opt(2)*ones(1,ns(2)) y_opt(3)*ones(1,ns(3)) y_opt(4)*ones(1,ns(4))];
    [error_th,error_th_task_init,error_emp_task_init,alpha_task_init, b_task,score_mean_task_init,variance_task,score_emp_task_init,var_emp_task_init,y_opt_init] = MLSSVRTrain_th1_centered_fixed_pro(X1,X2, yop', gamma1,gamma2, lambda,M,C,J,test1,testm1,test2,testm2,ones(nst(1),1),-ones(nst(2),1),ones(nst(3),1),-ones(nst(4),1),ns,'task',wanted(v),'general');
% yop=[y_opt(1)*ones(1,ns(1)) y_opt(2)*ones(1,ns(2)) y_opt(3)*ones(1,ns(3)) y_opt(1)*ones(1,ns(4))];
% [error]=@(x) perf_theorique_fixed_pro_2(X1,X2, x(3), x(1),x(2),M,C,ns,wanted(v),'general');
% options = struct('MaxFunctionEvaluations', 100);
% [param,error_opt_th]=fmincon(error,init,[-1 0 0;0 -1 0;0 0 -1],[0;0;0],[],[],[],[],[], options);
% lambda_opt=param(1);gamma_opt1=param(2); gamma_opt2=param(3);
% [error_th,error_th_task,error_emp_task,alpha_task, b_task,score_mean_task,variance_task,score_emp_task,var_emp_task,y_opt] = MLSSVRTrain_th1_centered_fixed_pro(X1,X2, yop', gamma_opt1,gamma_opt2, lambda_opt,M,C,J,test1,testm1,test2,testm2,ones(nst(1),1),-ones(nst(2),1),ones(nst(3),1),-ones(nst(4),1),ns,'task',wanted(v),'general');
% [error_th_init,error_th_task_init,error_emp_task_init,alpha_task_init, b_task,score_mean_task_init,variance_task,score_emp_task_init,var_emp_task_init,y_opt_init] = MLSSVRTrain_th1_centered_fixed_pro(X1,X2, yc', gamma1,gamma2, lambda,M,C,J,test1,testm1,test2,testm2,ones(nst(1),1),-ones(nst(2),1),ones(nst(3),1),-ones(nst(4),1),ns,'task',wanted(v),'general');
axi1(v)=error_emp_task(3);
axi2(v)=error_emp_task(4);
axi1_th(v)=error_th_task(3);
axi2_th(v)=error_th_task(4);
axi1_init(v)=error_emp_task_init(3);
axi2_init(v)=error_emp_task_init(4);
axi1_th_init(v)=error_th_task_init(3);
axi2_th_init(v)=error_th_task_init(4);
end
%plot(axi2)
%save('axes_update.mat', 'axi1', 'axi2', 'axi1_th', 'axi2_th', 'axi1_th_init', 'axi2_th_init', 'axi1_init', 'axi2_init')
sprintf('(%12f, %12f)', [axi1', axi2']')
sprintf('(%12f, %12f)', [axi1_th', axi2_th']')
sprintf('(%12f, %12f)', [axi1_th_init', axi2_th_init']')
sprintf('(%12f, %12f)', [axi1_init', axi2_init']')
figure
plot(axi1_th,axi2,'r-')
hold on
plot(axi1_th,axi2_th,'r--')
plot(axi1_th,axi2_init,'g-')
plot(axi1_th,axi2_th_init,'g--')
sprintf('(%12f, %12f)', [1-axi2', axi1']')
sprintf('(%12f, %12f)', [1-axi2_th', axi1_th']')
sprintf('(%12f, %12f)', [1-axi2_th_init', axi1_th_init']')
sprintf('(%12f, %12f)', [1-axi2_init', axi1_init']')