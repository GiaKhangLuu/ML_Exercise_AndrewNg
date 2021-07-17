function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%--------------COMPUTE COST---------------
y_hat = X * theta;
J = (1 / (2 * m)) * (y_hat - y)' * (y_hat - y);
theta_exclude_bias = theta(2:end);
reg_term = (lambda / (2 * m)) * theta_exclude_bias' * theta_exclude_bias;
J += reg_term;

%-------------COMPUTE GRAD-----------------
grad = (1 / m) * X' * (y_hat - y);
temp = theta;
temp(1) = 0;
reg_term = (lambda / m) * temp;
grad += reg_term;






% =========================================================================

grad = grad(:);

end
