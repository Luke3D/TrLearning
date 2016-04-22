%RF model where the output is weighted by trees which give better
%prediction on target data
clear
load TrainData.mat
%% split by brace type (use only C-brace data) for patients
% patientdata = load('trainData_patient.mat');
% patientdata = patientdata.trainingClassifierData;
% brace = cellfun(@(x) x(7:9),patientdata.subject,'UniformOutput', false);
% patientdata.brace = brace;
% Cbrind = strcmp(patientdata.brace,'Cbr');   %indices of Cbrace Data
% cData.subjectID = patientdata.subjectID(Cbrind);
% cData.activity = patientdata.activity(Cbrind);
% cData.features = patientdata.features(Cbrind,:);
% cData.featureLabels = patientdata.featureLabels;
% cData.sessionID = patientdata.sessionID(Cbrind);
%% train on all patients but one (SOURCE MODEL)
Nsubj = length(unique(cData.subjectID));

%coding activities with num and proportion of activities
activities = unique(cData.activity);
StateCodes = cell(length(activities),2);
StateCodes(:,1) = activities;
StateCodes(:,2) = num2cell(1:length(activities)); %sorted by unique

for i = 1:length(activities)
    actprop(i) = sum(double(strcmp(cData.activity,activities(i))))/length(cData.activity);
    %recode activities with numbers
    inds = strcmp(cData.activity,activities(i));
    cData.labels(inds) = i;
end
%show ratio of activities
figure
bar(actprop)
set(gca,'XTick',1:length(activities),'XTickLabel',activities)

%train on all subjects but one
indtr = cData.subjectID ~= 1;
Xtr = cData.features(indtr,:);
Ytr = cData.labels(indtr);

%for now use 1st subject as test
indte = cData.subjectID == 1;
Xte = cData.features(indte,:);
Yte = cData.labels(indte);

%split target data from test data (use 4th session as target data)
indtarget = cData.sessionID(indte) == 4;
Xtarget = Xte(indtarget,:); Ytarget = Yte(indtarget);
Xte = Xte(~indtarget,:); Yte = Yte(~indtarget);

%train a forest on training data
ntrees = 100;
opts = statset('UseParallel',1);
% disp(['training model ', num2str(s)])
RF = TreeBagger(ntrees,Xtr,Ytr,'Options',opts)

%compute accuracy of forest on remaining subject
[Yfit,P_RF] = predict(RF,Xte);
Yfit = str2num(cell2mat(Yfit));
cmat = confusionmat(Yte,Yfit)
accRF = trace(cmat)/sum(sum(cmat))
cmat_norm = cmat./repmat(sum(cmat,2),[1 length(cmat)])

%% reweigh the output based to favor trees with lower error rates
%compute accuracy for each tree
acc = [];
for t = 1:ntrees
    yt = RF.Trees{t}.predict(Xtarget);
    yt = str2num(cell2mat(yt));
    acc(t) = sum(yt==Ytarget')/length(Ytarget);
end
figure, plot(acc), xlabel('Tree'), ylabel('accuracy'), title('Accuracy on target data')

treeWeights = exp(1./(1-acc));
treeWeights = (treeWeights-min(treeWeights))/(max(treeWeights)-min(treeWeights))
figure, plot(treeWeights),xlabel('Tree'), ylabel('weight'), title('Weight on each tree')

[Yfit_target,P_RF_target]=predict(RF,Xte,'TreeWeights',treeWeights);
Yfit_target = str2num(cell2mat(Yfit_target));
cmat = confusionmat(Yte,Yfit_target)
accRF_target = trace(cmat)/sum(sum(cmat))
cmat_norm = cmat./repmat(sum(cmat,2),[1 length(cmat)])



