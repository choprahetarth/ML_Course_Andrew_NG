function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
minError = Inf;
currentC = Inf;
currentSigma = Inf;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

result = [100,100,100];
count = 0;
for C_values = [0.01 0.03 0.1 0.3 1 3 10 30]
  for sigma_values = [0.01 0.03 0.1 0.3 1 3 10 30]
    #fprintf('iteration number = %f', count++);
    count++;
    model = svmTrain(X, y, C_values, @(x1, x2) gaussianKernel(x1, x2, sigma_values)); 
    predictions = svmPredict(model,Xval);
    prediction_error = mean(double(predictions ~= yval));
    if prediction_error < minError
			minError = prediction_error;
			currentC = C_values;
			currentSigma = sigma_values;
    end
   end
end


C = currentC;
sigma = currentSigma;


% =========================================================================

end
