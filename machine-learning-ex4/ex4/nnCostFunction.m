function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%



X = [ones(m, 1) X];

Z2 = X*Theta1';

A2 = sigmoid(Z2);

A2 = [ones(size(A2, 1), 1)  A2];

Z3 = A2*Theta2';

A3 = sigmoid(Z3);


p = zeros(size(A3));

for i = 1:m
    
    [~, I] = max(A3(i,:));
    
    p(i, I) = 1;
    
    
    
end



y_true = zeros(size(A3));

for i = 1:m
    
    y_true(i, y(i)) = 1;
    
end

J = sum(sum(((-y_true.*log(A3)) - ((1-y_true).*log((1-A3))))))/m;

[~, cols1] = size(Theta1);

[~, cols2] = size(Theta2);


J = J + (lambda/(2*m))*(sum(sum(Theta1(:,2:cols1).^2)) + sum(sum(Theta2(:,2:cols2).^2)));



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.





delta_3 = A3-y_true;

delta_2 = Theta2'*delta_3';

delta_2 = delta_2(2:end, :);

delta_2 = delta_2.*sigmoidGradient(Z2)';

D_2 = delta_3'*A2;

D_1 = delta_2*X;

%delta2 needs to be 25 x 401
%delta3 needs to be 10 x 26


Theta2_grad = D_2/m;
Theta1_grad = D_1/m;

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + Theta2(:, 2:end)*lambda/m;  

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + Theta1(:, 2:end)*lambda/m;  





%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end