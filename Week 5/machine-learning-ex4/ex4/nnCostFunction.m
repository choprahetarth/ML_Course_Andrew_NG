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
%% num_labels is 10 
%% size (1,X) = unrolled 20*20 dimension vector 
% Setup some useful variables
m = size(X, 1); %% this is the number of rows
p = zeros(size(X, 1), 1);
         
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% PART 1 Forward Propagation
X = [ones(m,1) X];
A_1 = sigmoid(X*Theta1');
A_1=[ones(m,1) A_1];
Z_3 = A_1*Theta2';
h = sigmoid(Z_3);
%% h is now a 5000 * 400 matrix where it has 5000 examples and 400 being the total image unrolled

%% Modifying Y, because that will make proper use of [0,0,0,1,0,0,0,0,0,0] representing 3
%% other than  [10,1,2,3,4,5,6,7,8,9]
%% y new will be a 10 * 5000 matrix
y_new = eye(num_labels)(y,:); 
%% jaha Y me non zero value hai, usi column of y_new ke ye 1 value dal dega (sab above wala)

%% now calculate the cost
J = -(1/m)*sum(sum(y_new.*log(h)+(1-y_new).*(log(1-h))));
%% the double sum is used because the first sum function sums all the rows and the
%% second sum function is used to sum all the columns

%% PART 1.5 Regularization algorithm
J = J + (lambda/(2*m))*(sum(sum(Theta1.*Theta1))+sum(sum(Theta2.*Theta2))-(sum(Theta1(:,1).^2))-(sum(Theta2(:,1).^2)));
%% in the above code, simple algorithm is implemented, then the first theta values are removed
 
%% PART 2 Backpropagation code
%% completed the sigmoid gradient code
%% initiating the forward propagation
Z_2=X*Theta1';
a1 = zeros(size(X));
a2 = zeros(size(A_1));
a3 = zeros(size(h));
y_t = zeros(size(h));
z2 = zeros(size(Z_2));
del1 = zeros(size(Theta1));
del2 = zeros(size(Theta2));
%% part 2.2 perform forward propagation
for i = 1:m 
  a1(i,:) = X(i,:);
  a2(i,:) = A_1(i,:);
  a3(i,:) = h(i,:);
  y_t(i,:) = y_new(i,:);
  delta3 = a3 - y_t;
  z2(i,:) = Z_2(i,:);
  delta2 = delta3*Theta2 .* sigmoidGradient([1,z2(i,:)]);
  del1 = del1 + delta2.'*a1.';
	del2 = del2 + d3' * a2;
end;
fprintf('A ONE %f %f\n',size(a1));
fprintf('A TWO MATRIX %f %f\n',size(a2));
fprintf('A THREE MATRIX %f %f\n',size(a3));
fprintf('Y_T MATRIX %f %f\n',size(y_t));
fprintf('delta3 MATRIX %f %f\n',size(delta3));
fprintf('Theta1 MATRIX %f %f\n',size(Theta1));
fprintf('z2 MATRIX %f %f\n',size(z2));
fprintf('Theta2 MATRIX %f %f\n',size(Theta2));
fprintf('delta3 MATRIX %f %f\n',size(delta2));
Theta1_grad = 1/m * del1 + (lambda/m)*[zeros(size(Theta1, 1), 1) regTheta1];
Theta2_grad = 1/m * del2 + (lambda/m)*[zeros(size(Theta2, 1), 1) regTheta2];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
