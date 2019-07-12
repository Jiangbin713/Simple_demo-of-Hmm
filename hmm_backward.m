function [beta,Pb]=hmm_backward(state_tran,b)
 %% Description
 %%%%%%input£º
       %%%%%%%state_tran: state transitaion matrix
       %%%%%%%b:output probability density for each time frame and state
              %row: state   column: time
              
%%%%%%output£º
      %%%%%%%alpha: Backward likelihoods
      %%%%%%%Pf: Overall likelihood of the observations
      
      
%%  Matrix of state-transition probabilities
a = [state_tran( (2:end-1),(2:end-1) )];%processed state_tran for backward algorithm           


%% Backward algorithm
 T = length(b(1,:)); % time
 n1 = length(a(1,:)); %branch
 n2 = length(a(:,1)); %state_j 
 beta = zeros(T,n1);
 
 for i=1:n2
 beta(T,i) = state_tran(i+1,end)'; %initialing beta_T(i)
 end
 for t = -(T-1):-1  %time t
     for si = 1:n1 %branch si
             beta(-t,si)=sum(  ( a(si,1:n2) )'.*b(1:n1,-t+1).*( beta(-t+1,1:n2) )' ) ; %beta_t(i)=sum[a_ij * b_j(t+1) * beta_t+1(j)]
             
     end
 end
 
 Pb = 0;

 Pb = sum(  (state_tran(1,2:2+n2-1))' .* b(:,-t) .* (beta(1,:))'  ); %P(O|¦Ë)=sum[¦°_i * b_i * beta_1(i)]

 
 