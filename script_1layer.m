%% Training phase
%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 156;  % 156 input features
hidden_layer_size = 100;   % 25 hidden units
num_labels = 51;          % 51 labels, from 1 to 51   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('test.mat');
X_origin = X;
y_origin = y;
y_dir = zeros(size(y,1),1);
[dummy,y_dir] = max(y_origin,[],2);
X = X(1:3000,:);
y = y(1:3000,:);
X_test = X_origin(3001:4000,:);
y_test = y_origin(3001:4000,:);
y_test_dir = zeros(size(y_test,1),1);
[dummy,y_test_dir] = max(y_test,[],2);
X_val = X_origin(4001:end,:);
y_val = y_origin(4001:end,:);

m = size(X, 1);
% y_new = zeros(m, num_labels);
% for i = 1:m
%     y_new(i, y(i)) = 1;
% end
% y = y_new;

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
%initial_Theta3 = randInitializeWeights(hidden_layer_size(2), num_labels);

% Unroll parameters
%initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)];
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction2(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))): end), ...
                 num_labels, (hidden_layer_size + 1));
             
% Theta3 = reshape(nn_params(1 + end-num_labels*(hidden_layer_size(2)+1):end), ...
%                  num_labels, (hidden_layer_size(2) + 1));           


% Create "short hand" for the cost function to be minimized
costFunction2 = @(p) nnCostFunction2(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X2, y2, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost2] = fmincg(costFunction, nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))): end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));
pause;
% displayData(Theta2(:, 2:end));
% pause;
%displayData(Theta3(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

% pred = predict(Theta1, Theta2, Theta3, X);
pred = predict2(Theta1, Theta2, X);
% 
% fprintf('\nTraining Set Accuracy: %f\n', mean(mean(double(pred == y)) * 100));
% pause;
% 
%% Testing phase
% 
pred_t = predict2(Theta1, Theta2, X_test);
t1 = pred_t(:,1) == y_test_dir;
t2 = pred_t(:,2) == y_test_dir;
t3 = pred_t(:,3) == y_test_dir;
t4 = pred_t(:,4) == y_test_dir;
t5 = pred_t(:,5) == y_test_dir;
num_correct = sum(t1+t2+t3+t4+t5);
percent = num_correct/length(y_test_dir);
% 
% fprintf('\nTesting Set Accuracy: %f\n', mean(mean(double(pred_t == y_test)) * 100));
% pause;
% 
% %% Cross-validation phase
% 
% pred_val = predict(Theta1, Theta2, X_val);
% 
% fprintf('\nCross-validation Set Accuracy: %f\n', mean(mean(double(pred_val == y_val)) * 100));