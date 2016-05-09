%% Classifier comparison
clear all
load trainData

%%  With Classification Learner

%recode activities with numbers
activities = unique(cData.activity);
StateCodes = cell(length(activities),2);
StateCodes(:,1) = activities;
StateCodes(:,2) = num2cell(1:length(activities)); %sorted by unique

for i = 1:length(activities)
    inds = strcmp(cData.activity,activities(i));
    cData.labels(inds) = i;
end
cData.labels = cData.labels';

%global model
disp(unique(cData.subjectID)); %show available subj
subjtest = 1;                  %test subject
indtest = cData.subjectID ==subjtest;  
indtrain = ~indtest;

Xtrain = cData.features(indtrain,:);
Xtrain = [Xtrain cData.labels(indtrain)]; 

%split target data from test data (use 4th session as target data)
Xtest = cData.features(indtest,:);
Ytest = cData.labels(indtest);
indtarget = cData.sessionID(indtest) == 4;
Xtest = Xtest(~indtarget,:); Ytest = Ytest(~indtarget)';

classificationLearner

%% predict with the trained classifier on the test data
yfit = trainedClassifier.predictFcn(Xtest);
cmat = confusionmat(Ytest,yfit);
acc = trace(cmat)/sum(sum(cmat)) %overall accuracy
cmat_norm = cmat./repmat(sum(cmat,2),[1 length(cmat)])

%Balanced Accuracy
for c = 1:length(unique(Ytest))
    ic = find(Ytest == c);
    err(c) = sum(yfit(ic)~=Ytest(ic))/length(Ytest(ic));
end
BER = mean(err);    %Balanced error rate
Bacc = 1-BER        %Balanced accuracy


