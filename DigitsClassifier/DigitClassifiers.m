%% Train a Conditional Gaussian Classifier to recognize hand-written digits
% load the data
load('digits_data.mat');

% get the size of dataset
[nfeatures_train, nsamples_train, nclasses_train] = size(digits_train);

% compute the mean, shared variance, and shared standard diviation
uki = sum(digits_train,2) / nsamples_train;
sharedVar = sum(sum(sum((digits_train - uki).^2))) / (nfeatures_train*nsamples_train*nclasses_train);
stdDiv = sqrt(sharedVar);

% plot the shared std-div, and means
f1 = figure;
stdDivMessg = ['Pixel noise standard deviation = ', num2str(stdDiv)];
for i = 1:nclasses_train
    s = subplot(2, 5, i);
    classTag = ['#', num2str(i)];
    if i == 3
        imagesc( reshape( squeeze( uki(:,:,i) ),8,8 ).' ); 
        axis equal; axis off; colormap gray; title(s, {stdDivMessg;[];classTag}); 
    else
        imagesc( reshape( squeeze( uki(:,:,i) ),8,8 ).' ); 
        axis equal; axis off; colormap gray; title(s, classTag); 
    end
end

%% Train a Naive-Bayes Classifier to recognize hand-written digits
% convert the real-valued features into binary features by thresholding the copies of original data
digits_train_copy = digits_train;
digits_test_copy = digits_test;

digits_train_copy(  digits_train_copy > 0.5 )  = 1;
digits_train_copy(  digits_train_copy <= 0.5 )  = 0;
digits_test_copy(  digits_test_copy > 0.5 )  = 1;
digits_test_copy(  digits_test_copy <= 0.5 )  = 0;

% compute eta
eta = zeros(nfeatures_train, nclasses_train);
for i = 1:nfeatures_train
    for j = 1:nclasses_train
        % given a class j, the probability of i'th feature bi = 1
        eta(i,j) = nnz(digits_train_copy(i,:,j)) / nsamples_train; 
    end
end

% plot eta for all 10 classes
f2 = figure;
for i = 1:nclasses_train
    s = subplot(2, 5, i);
    classTag = ['#', num2str(i)];
    imagesc( reshape( eta(:,i),8,8 ).' ); axis equal; axis off; colormap gray; title(s, classTag); 
end

%% Hand-written digit recognition
[nfeatures_test, nsamples_test, nclasses_test] = size(digits_test);

%===================
% Gaussian Classfier
%===================
% compute p(x|Ck)
pxCk = zeros(nsamples_test*nclasses_test, nclasses_test);
for i = 1:(nsamples_test*nclasses_test)
    for j = 1:nclasses_test
        pxCk(i,j) = (2*pi*sharedVar)^(-nfeatures_test/2) * exp( -1/(2*sharedVar) * sum( ( digits_test(:,i) - uki(:,1,j) ).^2 ));
    end
end

% p(Ck|x) = p(x|Ck)*p(Ck)/p(x)
pCkx = pxCk * (1/nclasses_test) / (1/nsamples_test*nclasses_test);

% select the most likely class for each test cases
[~,I_gaussian] = max(pCkx,[],2);
I_gaussian = reshape ( I_gaussian,400,10 );

% count the number of misclassified cases for all 10 classes
numError_guassian = zeros(1, 10);
for i = 1:10
    numError_guassian(i)=nnz(I_gaussian(:,i)~=i);
end

% overall error rate in percentage
errorRate_gaussian = 100 * sum(numError_guassian) / (nsamples_test*nclasses_test);

%======================
% Naive-Bayes classfier
%======================
% compute p(Ck|b)
pCkb = zeros(nsamples_test*nclasses_test, nclasses_test);
digits_test_copy_complement = 1 - digits_test_copy;
for i = 1:(nsamples_test*nclasses_test)
    for j = 1:nclasses_test
        pCkb(i,j) = (1/nclasses_test) * prod( digits_test_copy(:,i) .* eta(:,j) + digits_test_copy_complement(:,i) .* (1 - eta(:,j)) );
    end
end

% select the most likely class for each test cases
[~,I_bayes] = max(pCkb,[],2);
I_bayes = reshape ( I_bayes,400,10 );

% count the number of misclassified cases for all 10 classes
numError_bayes = zeros(1, 10);
for i = 1:10
    numError_bayes(i)=nnz(I_bayes(:,i)~=i);
end

% overall error rate in percentage
errorRate_bayes = 100 * sum(numError_bayes) / (nsamples_test*nclasses_test);

% final error table for both classifiers
final_table = [ [numError_guassian, errorRate_gaussian]; [numError_bayes, errorRate_bayes] ];
disp(final_table);

















