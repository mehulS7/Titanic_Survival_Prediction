data = load('normalised_train_data.txt'); %loading Normalised train data
testdata = load('normalised_test_data.txt'); %loading Normalised test data

size(data) %number of rows and columns - in this last column is target column

% we have X and y here 
X = data(:,1:20);
% adding bias coefficient
X = [ones(length(X),1) X];                                                 
y = data(:,21);

% Xtest for testdata
% adding bias coefficient
Xtest = [ones(length(testdata),1) testdata];                                         

% Defining our hypothesis for logistic regresion
function [h] = hypothesis(X,theta)
z = X*theta;
h = 1 ./ (1 + e.^(-z));                                             
end

% Defining our function to optimise theta with gradient boosting
function [theta, J] = grad(X,y,alpha,num_iter)
m = length(y);                                                             % m = number of observation
theta = rand(size(X,2) ,1);                                                % randomly initializing theta at first
for j=1:num_iter                                                           % loop to iterate gradient to optimize theta values  
h = hypothesis(X, theta);                                                  % getting our hypothesis                                         
J(j) = -((1/m) * (sum((y .* log(h)) + ((1-y) .* log(1 - h)))));        % Cost function calculation and storing values for later observation
grad = (alpha/m) .* (sum((h - y) .* X));                                   % our gradient calculation
theta = theta - grad';                                                      % optimised theta 
end
end


%Defining prediction function
function [p] = pred(X,theta)
h = hypothesis(X,theta);
p = h>=0.5;                       %probabolity comes greater than 0.5 it shall be treated as Survived = 1
end

%Running gradient algorithm
[theta, J] = grad(X,y,0.5,1000);    %optimised theta is returned and cost value for each iteration too

%plot to see how gradient algorithm worked
plot(1:length(J),J);

%Running prediction function
[p] = pred(Xtest,theta);

%prediction Accuracy @Kaggle was .76555