
 function [alpha,Pf]=hmm_forward(state_tran,b)
 %% Description
 %%%%%%input£º
       %%%%%%%state_tran: state transitaion matrix
       %%%%%%%b:output probability density for each time frame and state
              %row: state   column: time
              
%%%%%%output£º
      %%%%%%%alpha: Forward likelihoods
      %%%%%%%Pf: overall likelihood of the observations
      
      
%% State translation aij
a = state_tran((2:end-1),(2:end-1));   %processed state translation for forward algorithm    


 
 %% Forward algorithm
 T = length(b(1,:)); %time 
 n1 = length(a(1,:)); %branch
 n2 = length(a(:,1)); %state_j 
 
 alpha = zeros(T,n1); %initialing alpha matrix
 
 for i= 1:n1  
     alpha(1,i)=state_tran(1,1+i)*b(i,1); %initialing alpha a1(1) a1(2) a1(3) ¦°i * bi(o1)
 end
 for t = 2:T  %time t
     for si = 1:n1 %branch si
           
          alpha(t,si)=sum( alpha(t-1,1:n2).*(a(1:i,si))' ) * b(si,t); %alpha_t(i)=sum[a_t-1(i)*alpha(ij)]*bj(t)
          
     end
 end

Pf = 0; %overall forward posibility
for i=1:n2
    Pf = Pf+alpha(t,i)*state_tran(1+i,end); %P(O|¦Ë)=sum[alpha_T(i)]
end
 