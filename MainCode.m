%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%2018 assignment of Speech & Audio Processing & Recognition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%BIN JIANG 6519680
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Important Variants:
%alpha 
%beta  
%epsilon 
%gama  
%Pf : Forward probability
%Pb : Backward probability
%state_tran: state transition
%u: re-estimated mean
%var : re-estimated variance
%A : A matrix
%b : pdfs
%B : re-estimated pdfs
%u_p : previous mean
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Fuction:
%hmm_forward.m: calculate forward probability
%hmm_backward.m: calculate backward probability
%ReEstimate.m: re-estimate mean and variance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clear all
%% 1.Plots of state topology 
   figure (1)
   ST=imread('State topology.png');
   imshow (ST);
   title('State topology');

%% 1.Plot the output probability density function(pdfs)
step = 0.05;
x = -2:step:8; % setting step interval
u_p =[1.9;3.4;5.1];
state_1=normpdf(x,u_p(1),sqrt(0.16)); %output probability density state 1
state_2=normpdf(x,u_p(2),sqrt(0.81)); %output probability density state 2
state_3=normpdf(x,u_p(3),sqrt(0.25)); %output probability density state 3

figure (2)    %plot 
plot(x,state_1)
hold on
plot(x,state_2)
plot(x,state_3,'k')

title('Output probability density functions');
xlabel('O');ylabel('P(O)');
legend('state 1','state 2','state 3');
% c = 'C:\Users\jiang\Desktop\CVML\Sem1-Speech & Audio Processing & Recognition\Coursework2-JAN 8 TUE';
% saveas(gcf,['c','pdfs','jpeg'])

%% 2.Matrix of state-transition probabilities

state_tran = [0  0.93   0.07      0     0;
              0  0.84   0.11   0.05     0;
              0     0   0.88   0.08  0.04;
              0     0      0   0.91  0.09;
              0     0      0      0     0;]; % giving state transition matrix
          
%% 2.Output probability density for each time frame and state

O = [1.3, 2.3, 2.8, 3.3, 5.0, 5.6, 4.9, 5.9]; %oberservations
b1=zeros(1,8);
b2=zeros(1,8);
b3=zeros(1,8);

for k = 1:8
b1(k) = ( 1/( sqrt(0.16)*sqrt(2*pi) ) ) *exp( (-(O(k)-1.9)^2)/ ( 2*0.16 ) );
b2(k) = ( 1/( sqrt(0.81)*sqrt(2*pi) ) ) *exp( (-(O(k)-3.4)^2)/ ( 2*0.81 ) );
b3(k) = ( 1/( sqrt(0.25)*sqrt(2*pi) ) ) *exp( (-(O(k)-5.1)^2)/ ( 2*0.25 ) );
end

b =[ b1;    %output probability density for each time frame and state
     b2;    %row: state
     b3;];  %column: time
 
 fprintf('Output probability density functions b_i(o_t):\n %');
 fprintf('Row(state)  Column(time)\n %');
 disp(b); %% print the output probability densities b_i(o_t) for each time frame and state;

%% 3.Forward likelihoods and overall likelihood of the observations
[alpha,Pf] = hmm_forward(state_tran,b); % alpha: forward likelihoods 
                                        % Pf: overall likelihood of the observations

fprintf('Overall likelihood of the observations: %e (Forward)\n', Pf)       

fprintf('Forward likelihoods of the observation:\n')   
fprintf('Row(time) Column(state)\n')
[m,n]=size(alpha);

for i = 1:m               %output alpha
    for j=1:n
        fprintf('%e    ',alpha(i,j));
    end
    fprintf('\n');
end
fprintf('\n');

%% 4.Backward likelihoods and overall likelihood of the observations
[beta,Pb] = hmm_backward(state_tran,b); % beta: backward likelihoods 
                                        % Pf: overall likelihood of the observations

fprintf('Overall likelihood of the observations: %e  (Backward)\n', Pb)       

fprintf('Backward likelihoods of the observation:\n')   
fprintf('Row(time) Column(state)\n')
[m,n]=size(beta);


for i = 1:m               %output beta
    for j=1:n
        fprintf('%e    ',beta(i,j));
    end
    fprintf('\n');
end
fprintf('\n');     

%% 5.Occupation likelihoods
gama = (alpha.*beta)./Pf; %¦Ã_t(i)= [¦Á_t(i)*¦Â_t(i)]/P(O|¦Ë)

fprintf('Occupation likelihood:\n')   
fprintf('Row(time) Column(state)\n')
[m,n]=size(gama);

for i = 1:m             %output gama
    for j=1:n
        fprintf('%e    ',gama(i,j));
    end
    fprintf('\n');
end
fprintf('\n');     

%% 6.Re-estimated means and variances
[x,y]=size(b);
B = zeros(x,y);

 for j = 1: length(b(:,1))             
     for k = 1: length(b(1,:))
         B(j,k)= gama(k,j) / sum( gama(:,j) );  %Re-estimated output probability density functions
     end
 end

[u,var] = ReEstimate(B,O,gama,u_p); %Re-estimated mean and variance

x = -2:step:8;
state_11= normpdf(x,u(1),sqrt(var(1)) );  %plot normal distribution
state_22= normpdf(x,u(2),sqrt(var(2)) );
state_33= normpdf(x,u(3),sqrt(var(3)) );
plot(x,state_11,'--')
plot(x,state_22,'--')
plot(x,state_33,'--')
legend('State 1','State 2','State 3','State 1 £¨training£©','State 2 £¨training£©','State 3 £¨training£©');
c = 'C:\Users\jiang\Desktop\CVML\Sem1-Speech & Audio Processing & Recognition\Coursework2-JAN 8 TUE';
saveas(gcf,['normal','jpeg'])
% for i = 1: length(b(:,1))
%     
%     u(i) = sum( gama(:,i).*B(i,:)' ) / sum(gama(:,i));
%     
%     variance(i) = sum( gama(:,i)'.*( B(i,:)-u(i) ).^2) / sum(gama(:,i));
% 
% end

% for i = 1:length(b(:,1))
%      
%     u(i) = sum (B(i,:))/length(b(:,1))
% 
% end


%% 7. Plots of the pdfs and comments
fprintf('Output probability density functions (after training) b_i(o_t):\n %');
fprintf('Row(state)  Column(time)\n %');
disp(B); %% print the output probability densities b_i(o_t) for each time frame and state;
%% 8. Re-estimation transition likelihood
[x,y]=size(state_tran);
  
x=x-2;
y=y-2; %processed size of real transition matrix
T = length(b(1,:));
epsilon=zeros(x,y,T); % x £ºi   y £ºj   z£ºt
a = [state_tran( (2:end-1),(2:end-1) )];%processed state_tran 
p=Pf; %forward or backward probability
 for t = 2:T
     for i = 1:x
         for j= 1:y
             epsilon(i,j,t)= (alpha(t-1,i)*a(i,j)*b(j,t)*beta(t,j)) / p; %epsilon
         end
     end
 end
 
fprintf('Re-estimation transition likelihood:\n')   
fprintf('Row(i) Column(j)\n')
[m,n,~]=size(epsilon);

for t = 1:T            %print epsilon
    fprintf('T=%d\n',t) ; 
    for i=1:m
        for j=1:n
          
        fprintf('%e    ',epsilon(i,j,t));
        end  
    fprintf('\n');
    end
fprintf('\n')
end

 %% 9. Re-estimated A matrix
 A=zeros(x+2,y+2); 
    for i =1:x
        for j=1:y
        A(i+1,j+1)= sum( epsilon(i,j,2:T) ) / sum( gama(:,i) );   %Re-estimated A matrix
        end
    end
A(1,2:2+length(gama(1,1:end))-1) = gama(1,:);
A(3,end)=1-sum(A(3,:));
A(4,end)=1-sum(A(4,:));
fprintf('Re-estimated A matrix:\n')   
fprintf('Row(i) Column(j)\n')
[m,n]=size(A);

for i = 1:m     %print
    for j=1:n
        fprintf('%e    ',A(i,j));
    end
    fprintf('\n');
end
fprintf('\n'); 
    