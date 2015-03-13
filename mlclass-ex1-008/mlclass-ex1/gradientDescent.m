function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2); % number of parameters
J_history = zeros(num_iters, 1);
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    for iter = 1:num_iters,
       delta = zeros(n,1);
       for i=1:m,
          delta = delta + (theta'*X(i,:)'-y(i))*X(i,:)';
       end;
       theta = theta - alpha/m*delta;
    % ============================================================

    % Save the cost J in every iteration    
       J_history(iter) = computeCost(X, y, theta);  
    end

end
