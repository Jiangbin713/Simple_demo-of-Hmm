function [u,var]=ReEstimate(B,O,gama,u_p)
 %% Description
 %%%%%%input£º
       %%%%%%%B: Re-estimated output probability density functions
       %%%%%%%O: Obeservation sequences
       %%%%%%%gama: Occupation likelihoods
       %%%%%%%u_p: previous mean
              
%%%%%%output£º
      %%%%%%%u: Re-estimated mean
      %%%%%%%var: Re-estimated variance
      
%step = 0.05; % plot interval

%n = (length(B(1,:)) / step)+1; % total data    (It can be any number)

%OO = repmat(O,[3,1]); %OO = [O;O;O];

%nn = round(n*B);  % frequence of each observations in each state

for i = 1:length(B(:,1))
    u(i) = sum( gama(:,i).*O' ) / sum(gama(:,i));  
    var(i)= sum(gama(:,i).*(O'-u_p(i)).^2)/sum(gama(:,i));
end


%uc = repmat(u_p,[1,length(B(1,:))]); % 

%var=
%var = sum((((OO-uc).^2).*OO)')/n;
