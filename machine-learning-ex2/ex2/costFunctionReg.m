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

% Compute cost
[J, grad] = costFunction(theta, X, y);
theta_exclude_bias = theta(2:length(theta));
reg_term = (lambda / (2 * m)) * theta_exclude_bias' * theta_exclude_bias;
J = J + reg_term;

% Compute gradient
grad(2:length(grad)) = grad(2:length(grad)) + (lambda / m * theta_exclude_bias);

% =============================================================

end
