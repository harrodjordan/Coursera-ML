function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

[length_X, width] = size(X);

for i = 1:length_X
    
    theta_trans = theta';
    x_i= X(i,:);
    y_i = y(i);
    
    temp = (-y_i*log(sigmoid(dot(theta_trans,x_i)))) - ((1-y_i)*log(1-sigmoid(dot(theta_trans,x_i))));
    
    J = J+temp;
    
    for j = 1:width
    
        temp2 = (sigmoid(dot(theta_trans,x_i)) - y_i)*X(i,j);
        grad(j) = grad(j) + temp2;
        
    end
    
    
  
end


J = J/length_X;
grad = grad./length_X;









% =============================================================

end
