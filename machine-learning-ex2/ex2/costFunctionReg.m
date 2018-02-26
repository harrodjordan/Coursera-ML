function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


[length_X, width] = size(X);

J_theta = 0;

for i = 1:length_X
    
    theta_trans = theta';
    x_i= X(i,:);
    y_i = y(i);
    
    temp = (-y_i*log(sigmoid(dot(theta_trans,x_i)))) - ((1-y_i)*log(1-sigmoid(dot(theta_trans,x_i))));
    
    J = J+temp;
    
    for j = 1:width
        
        temp2 = (sigmoid(dot(theta_trans,x_i)) - y_i)*X(i,j);
        grad(j) = grad(j) + temp2;
        
        
        if i == 2
        
            temp3 = theta_trans(j)^2;
            J_theta = J_theta + temp3;
            
        end
        
    end
    
    
  
end

J = J/length_X + (lambda*J_theta/(2*length_X));
grad = grad./length_X;

for k = 2:length(grad)
    
    grad(k) = grad(k) + (lambda/length_X)*theta(k);
    
end




% =============================================================

end
