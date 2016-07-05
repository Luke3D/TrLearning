%export data from healthy and patients
load trainData_healthy.mat

% HealthyData = table();
% % var1 = 'SubjID';
% var2 = 'Session';
% var3 = 'Features';
% var4 = 'Label';

SubjID = trainingClassifierData.subjectID;
Session = trainingClassifierData.sessionID;
Features = trainingClassifierData.features;

%convert activity labels to numeric codes
activities = unique(trainingClassifierData.activity);
StateCodes = cell(length(activities),2);
StateCodes(:,1) = activities;
StateCodes(:,2) = num2cell(0:length(activities)-1); %sorted by unique

for i = 1:length(activities)
    %recode activities with numbers
    inds = strcmp(trainingClassifierData.activity,activities(i));
    Label(inds) = i-1;
end
Label = Label';

Healthydata = table(SubjID,Session,Features,Label);

writetable(Healthydata,'./Export/HealthyData.csv')

%% Patient CBR data
clear all
load trainDataCBR.mat

%convert activity labels to numeric codes
activities = unique(trainingClassifierData.activity);
StateCodes = cell(length(activities),2);
StateCodes(:,1) = activities;
StateCodes(:,2) = num2cell(0:length(activities)-1); %sorted by unique

for i = 1:length(activities)
    %recode activities with numbers
    inds = strcmp(trainingClassifierData.activity,activities(i));
    Label(inds) = i-1;
end
Label = Label';

SubjID = trainingClassifierData.subjectID;
Session = trainingClassifierData.sessionID;
Features = trainingClassifierData.features;

CBRdata = table(SubjID,Session,Features,Label);

writetable(CBRdata,'./Export/PatientCBRData.csv')


%% SCO data
clear all
load trainDataSCO.mat

%convert activity labels to numeric codes
activities = unique(trainingClassifierData.activity);
StateCodes = cell(length(activities),2);
StateCodes(:,1) = activities;
StateCodes(:,2) = num2cell(0:length(activities)-1); %sorted by unique

for i = 1:length(activities)
    %recode activities with numbers
    inds = strcmp(trainingClassifierData.activity,activities(i));
    Label(inds) = i-1;
end
Label = Label';

SubjID = trainingClassifierData.subjectID;
Session = trainingClassifierData.sessionID;
Features = trainingClassifierData.features;

SCOdata = table(SubjID,Session,Features,Label);

writetable(SCOdata,'./Export/PatientSCOData.csv')
