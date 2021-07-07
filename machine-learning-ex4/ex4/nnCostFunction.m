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

% --------PART 1--------

% Feedforward
X = [ones(m, 1), X];
A_1 = X;
Z_2 = X * Theta1';
A_2 = sigmoid(Z_2);
A_2 = [ones(m, 1), A_2];
Z_3 = A_2 * Theta2';
A_3 = sigmoid(Z_3);

% Compute cost
for i = 1:num_labels
        curr_label = (y == i);
        curr_probs = A_3(:, i);
        curr_cost = sum(curr_label .* log(curr_probs) + (1-curr_label) .* log(1 - curr_probs));
        J += curr_cost;
end
J *= -1 / m;

% --------PART 2--------
% Unregularization
delta_1 = 0;
delta_2 = 0;
for i = 1:m

        % Feedforward
        a_1 = X(i, :)';
        curr_y = zeros(num_labels, 1);
        curr_y(y(i)) = 1; 
        z_2 = Theta1 * a_1;
        a_2 = sigmoid(z_2);
        a_2 = [1; a_2];
        z_3 = Theta2 * a_2;
        a_3 = sigmoid(z_3);

        % Backprop
        err_3 = a_3 - curr_y;
        err_2 = Theta2(:, 2:end)' * err_3 .* sigmoidGradient(z_2);

        % Add up delta
        delta_1 += err_2 * a_1';
        delta_2 += err_3 * a_2';
end
Theta1_grad = (1 / m) * delta_1;
Theta2_grad = (1 / m) * delta_2;

% Regularization
temp_1 = Theta1;
temp_2 = Theta2;
temp_1(:, 1) = zeros(size(Theta1, 1), 1);
temp_2(:, 1) = zeros(size(Theta2, 1), 1);
Theta1_grad += (lambda / m * temp_1);
Theta2_grad += (lambda / m * temp_2);

% --------PART 3--------
% Regularization
Theta1_exclude_bias = Theta1(:, 2:end);
Theta2_exclude_bias = Theta2(:, 2:end);
regu_term = lambda / (2 * m) * (sum(Theta1_exclude_bias(:) .^ 2) + sum(Theta2_exclude_bias(:) .^ 2));
J += regu_term;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
