function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


    
for k = 1:m 
    
%     j = length(theta);
%     old_theta = theta;
%     
%     for update = 1:j
%         
%         sum = zeros(j,1);
%         
%         for i = 1:m
%             sum(update) = sum(update) + (X(i,update))*(sigmoid(X(i,:)*old_theta) - y(i));
%             
%         end
%           theta(update) = theta(update) - ((1/m)*sum(update,1));
%           
%     end
    J = J + ((-1*y(k)*log(sigmoid(X(k,:)*theta))) - ((1-y(k))*log(1 - sigmoid(X(k,:)*theta))));
    disp(J)
end

    J = J/m;


% =========================================================================

end
