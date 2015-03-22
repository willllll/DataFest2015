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
Theta1 = reshape(nn_params(1:hidden_layer_size(1) * (input_layer_size + 1)), ...
                 hidden_layer_size(1), (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size(1) * (input_layer_size + 1))): end - num_labels * (hidden_layer_size(2) + 1)), ...
                 hidden_layer_size(2), (hidden_layer_size(1) + 1));
             
Theta3 = reshape(nn_params(1 + end-num_labels*(hidden_layer_size(2)+1):end), ...
                 num_labels, (hidden_layer_size(2) + 1));           

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

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

%% Part 1.
X = [ones(m,1) X];
z2 = X * Theta1';
a2 = sigmoid(z2);

a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

a3 = [ones(m,1) a3];
z4 = a3 * Theta3';
a4 = sigmoid(z4);
h_theta = a4;

%y_new = zeros(m, num_labels);
y_new = y;
for i = 1:m
    %y_new(i, y(i)) = 1;
    J = J + 1/m * (-log(h_theta(i,:)) * y_new(i,:)' - log(1 - h_theta(i,:)) * (1 - y_new(i,:))');
end
%J = 1/m * sum(sum(-log(h_theta) .* y_new - log(1 - h_theta) .* (1 - y_new)));

J = J + lambda / (2 * m) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));

%% Part 2.

% Set up Delta matrix to accumulate gradients
Delta1 = zeros(hidden_layer_size(1), input_layer_size + 1);
Delta2 = zeros(hidden_layer_size(2), hidden_layer_size(1) + 1);
Delta3 = zeros(num_labels, hidden_layer_size(2) + 1);

% Loop through all observations
for i = 1:m
    % Forward propagate each observation to obtain deltas for each layer
    a1 = X(i,:)';
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1;a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    a3 = [1;a3];
    z4 = Theta3 * a3;
    a4 = sigmoid(z4);
    
    delta_4 = a4 - y_new(i,:)';
    delta_3 = (Theta3(:,2:end)' * delta_4) .* sigmoidGradient(z3);
    delta_2 = (Theta2(:,2:end)' * delta_3) .* sigmoidGradient(z2);
    
    % Accumulate Delta
    Delta1 = Delta1 + (delta_2 * a1');
    Delta2 = Delta2 + (delta_3 * a2');
    Delta3 = Delta3 + (delta_4 * a3');
end

% Compute gradient
Theta1_grad = 1/m * Delta1;
Theta2_grad = 1/m * Delta2;
Theta3_grad = 1/m * Delta3;

%% Part 3.

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);
Theta3_grad(:, 2:end) = Theta3_grad(:, 2:end) + lambda / m * Theta3(:, 2:end);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end