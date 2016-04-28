%% DIRECTORIES
clear all, close all;

patient_stairs = [2 8 11 12 14 15];
disp(patient_stairs);

p = gcp('nocreate');
if isempty(p)
    parpool('local')
end

cd(fileparts(which('PatientStacked.m')))
currentDir = pwd;
slashdir = '/';
addpath([pwd slashdir 'sub']); %create path to helper scripts
addpath(genpath([slashdir 'Traindata'])); %add path for train data

plotON = 1;                             %draw plots
drawplot.activities = 0;                % show % of each activity
drawplot.accuracy = 0;
drawplot.actvstime = 0;
drawplot.confmat = 0;

%Additional options
clipThresh = 0; %to be in training set, clips must have >X% of label
OOBVarImp = 'off';   %enable variable importance measurement

%% LOAD DATA TO ANALYZE
population = 'patient';
filename = ['trainData_' population '.mat'];
load(filename)

%% SELECT PATIENT
tt = num2str(unique(trainingClassifierData.subjectID)');
fprintf('\n')
fprintf('Subject IDs present for analysis: %s',tt)
fprintf('\n')
fprintf('Available files to analyze: ')
fprintf('\n')
disp(unique(trainingClassifierData.subject))
fprintf('\n')

all_subjectID = trainingClassifierData.subjectID;

proceed = 1;
while proceed > 0 
    subject_analyze = input('Subject ID to analyze (ex. 5): ');

    %Check if subjectID is in mat file
    if ~any(subject_analyze == all_subjectID)
        disp('-------------------------------------------------------------')
        disp('Subject ID not in trainingClassifierData.mat file. Try again.')
        disp('-------------------------------------------------------------')
    else
        subject_indices = find(subject_analyze==all_subjectID);
        proceed = 0;
    end
end

cData_temp2 = isolateSubject(trainingClassifierData,subject_indices);

if strcmpi(population,'patient')
    for zz = 1:length(cData_temp2.subject)
        temp = char(cData_temp2.subject(zz));
        cData_temp2.subjectBrace(zz) = {temp(7:9)};
    end

    proceed = 1;
    while proceed > 0
        fprintf('\n')
        brace_analyze = input('Brace to analyze (SCO, CBR, both): ','s');

        %Check if brace entered is SCO or CBR or both
        if ~(strcmpi(brace_analyze,'SCO') || strcmpi(brace_analyze,'CBR') || strcmpi(brace_analyze,'BOTH'))
            disp('---------------------------------------------------------------')
            disp('Please correctly select a brace (SCO, CBR, or both). Try again.');
            disp('---------------------------------------------------------------')
        else
            %Check if SCO or CBR are in mat file
            if (strcmpi(brace_analyze,'both'))
                brace_analyze = 'both';

                if isempty(strmatch('Cbr',cData_temp2.subjectBrace)) || isempty(strmatch('SCO',cData_temp2.subjectBrace))
                    disp('--------------------------------------------------------')
                    disp('Brace not in trainingClassifierData.mat file. Try again.')
                    disp('--------------------------------------------------------')
                else
                    proceed = 0;
                end
            elseif (strcmpi(brace_analyze,'CBR'))
                brace_analyze = 'Cbr';

                if isempty(strmatch('Cbr',cData_temp2.subjectBrace))
                    disp('------------------------------------------------------')
                    disp('CBR not in trainingClassifierData.mat file. Try again.')
                    disp('------------------------------------------------------')
                else
                    proceed = 0;
                end
            elseif (strcmpi(brace_analyze,'SCO'))
                brace_analyze = 'SCO';

                if isempty(strmatch('SCO',cData_temp2.subjectBrace))
                    disp('------------------------------------------------------')
                    disp('SCO not in trainingClassifierData.mat file. Try again.')
                    disp('------------------------------------------------------')
                else
                    proceed = 0;
                end
            end
        end
    end

    cData_temp = isolateBrace(cData_temp2,brace_analyze);
else
    cData_temp = cData_temp2;
end

proceed = 1;
while proceed > 0
    fprintf('\n')
    disp('Please enter the max number of sessions to analyze.'); 
    disp('Or type 0 to analyze all sessions available.')
    min_sessions = input('Min session ID: ');
    max_sessions = input('Max session ID: ');
    proceed = 0;
end

if max_sessions == 0
    cData = cData_temp;
else
    cData = isolateSession(cData_temp,max_sessions,min_sessions);
end

fprintf('\n')
disp('These are the subjects that will be analyzed: ')
disp(unique(cData.subject))
fprintf('\n')

%% EXTRACT MAIN SESSIONS (personal data)
states = {'Sitting';'Stairs Dw';'Stairs Up';'Standing';'Walking'};

features_main     = cData.features; %features for classifier
subjects_main     = cData.subject;  %subject number
uniqSubjects_main = unique(subjects_main); %list of subjects
statesTrue_main = cData.activity;     %all the classifier data
subjectID_main = cData.subjectID;
sessionID_main = cData.sessionID;

%Remove stairs data from specific patients
stairs_remove = [];
for h = 1:length(patient_stairs)
    a1 = find(subjectID_main == patient_stairs(h));
    a2 = strmatch('Stairs Up',statesTrue_main,'exact');
    a = intersect(a1,a2);
    
    b1 = find(subjectID_main == patient_stairs(h));
    b2 = strmatch('Stairs Dw',statesTrue_main,'exact');
    b = intersect(b1,b2);
    
    stairs_remove = [stairs_remove; a; b];
end
features_main(stairs_remove,:) = [];
subjects_main(stairs_remove) = [];
statesTrue_main(stairs_remove) = [];
subjectID_main(stairs_remove) = [];
sessionID_main(stairs_remove) = [];
uniqStates_main  = unique(statesTrue_main); 

%Generate codesTrue
codesTrue_main = zeros(1,length(statesTrue_main));
for i = 1:length(statesTrue_main)
    codesTrue_main(i) = find(strcmp(statesTrue_main{i},states));
end
disp('Data extracted for sessions of interest.')

%% EXTRACT NEW SESSION (personal data)
%Isolate data
sessionID_train = 4;
subject = isolateSession(cData_temp,sessionID_train,sessionID_train);

%Extract data
features_new     = subject.features; %features for classifier
subjects_new     = subject.subject;  %subject number
uniqSubjects_new = unique(subjects_new); %list of subjects
statesTrue_new = subject.activity;     %all the classifier data
subjectID_new = subject.subjectID;
sessionID_new = subject.sessionID;

%Remove stairs data from specific patients
stairs_remove = [];
for h = 1:length(patient_stairs)
    a1 = find(subjectID_new == patient_stairs(h));
    a2 = strmatch('Stairs Up',statesTrue_new,'exact');
    a = intersect(a1,a2);
    
    b1 = find(subjectID_new == patient_stairs(h));
    b2 = strmatch('Stairs Dw',statesTrue_new,'exact');
    b = intersect(b1,b2);
    
    stairs_remove = [stairs_remove; a; b];
end
features_new(stairs_remove,:) = [];
subjects_new(stairs_remove) = [];
statesTrue_new(stairs_remove) = [];
subjectID_new(stairs_remove) = [];
sessionID_new(stairs_remove) = [];
uniqStates_new  = unique(statesTrue_new);

%Generate codesTrue
codesTrue_new = zeros(1,length(statesTrue_new));
for i = 1:length(statesTrue_new)
    codesTrue_new(i)  = find(strcmp(statesTrue_new{i},states));
end
disp('Data extracted for new session.')
fprintf('\n')

%% LAYER 1: TRAIN CLASSIFIERS + GENERATE POSTERIORS (PATIENTS)
%Isolate brace and sessions
for zz = 1:length(trainingClassifierData.subject)
    temp = char(trainingClassifierData.subject(zz));
    trainingClassifierData.subjectBrace(zz) = {temp(7:9)};
end
patients_temp_2 = isolateBrace(trainingClassifierData,brace_analyze);
patients_temp = isolateSession(patients_temp_2,max_sessions,min_sessions);

IDs = unique(trainingClassifierData.subjectID);
ind_temp = find(subject_analyze == IDs); %find patient to analyze
IDs(ind_temp) = []; %remove patient so list contains global patients

n_act = zeros(length(IDs),1);
n_train = zeros(length(IDs),1);

%Bias model variables
ntrees = 100;
accuracies = zeros(ntrees,length(IDs));
codesTree = cell(length(IDs),1);

disp('Initiating Layer 1...')
disp('Cycling through each global patient:')
%Cycle through each patient
for y = 1:length(IDs)
    
    %Allocate space
    codesTree{y} = zeros(length(codesTrue_main),ntrees);
    
    %Isolate  subject
    subject_indices = find(IDs(y)==patients_temp.subjectID);
    patients = isolateSubject(patients_temp,subject_indices);
    
    %Extract data
    features_p     = patients.features; %features for classifier
    subjects_p     = patients.subject;  %subject number
    uniqSubjects_p = unique(subjects_p); %list of subjects
    statesTrue_p = patients.activity;     %all the classifier data
    subjectID_p = patients.subjectID;
    sessionID_p = patients.sessionID;
    
    %Remove stairs data from specific patients
    stairs_remove = [];
    for h = 1:length(patient_stairs)
        a1 = find(subjectID_p == patient_stairs(h));
        a2 = strmatch('Stairs Up',statesTrue_p,'exact');
        a = intersect(a1,a2);
        
        b1 = find(subjectID_p == patient_stairs(h));
        b2 = strmatch('Stairs Dw',statesTrue_p,'exact');
        b = intersect(b1,b2);
        
        stairs_remove = [stairs_remove; a; b];
    end
    features_p(stairs_remove,:) = [];
    subjects_p(stairs_remove) = [];
    statesTrue_p(stairs_remove) = [];
    subjectID_p(stairs_remove) = [];
    sessionID_p(stairs_remove) = [];
    uniqStates_p  = unique(statesTrue_p);
    
    n_act(y) = length(uniqStates_p);
    n_train(y) = size(features_p,1);
    
    %Generate codesTrue
    codesTrue_p = zeros(1,length(statesTrue_p));
    for i = 1:length(statesTrue_p)
        codesTrue_p(i)  = find(strcmp(statesTrue_p{i},states));
    end
    
    %Train Random Forest on Global Patient
    disp(['RF Train - Patient '  num2str(IDs(y)) '  #Samples Train = ' num2str(size(features_p,1))]);
    opts_ag = statset('UseParallel',1);
    RFmodel_p = TreeBagger(ntrees,features_p,codesTrue_p,'OOBVarImp',OOBVarImp,'Options',opts_ag);
    
    %Predict with Each Tree from RF
    for p = 1:ntrees
        %Predict with each tree on fourth session
        codesTree_new = RFmodel_p.Trees{p}.predict(features_new);
        codesTree_new = str2num(cell2mat(codesTree_new));
        [~, accuracies(p,y)] = confusionMatrix_5(codesTrue_new,codesTree_new);
        
        %Predict with each tree on first three sessions
        codesTree_main = RFmodel_p.Trees{p}.predict(features_main);
        codesTree{y}(:,p) = str2num(cell2mat(codesTree_main));        
    end
end

%% LAYER 1: TRAIN CLASSIFIERS + GENERATE POSTERIORS (HEALTHY)
%Load healthy data
clear trainingClassifierData
population = 'healthy';
filename = ['trainData_' population '.mat'];
load(filename)

IDs_healthy = unique(trainingClassifierData.subjectID);

%Initialize Variables
n_act_h = zeros(length(IDs_healthy),1);
n_train_h = zeros(length(IDs_healthy),1);

accuracies_h = zeros(ntrees,length(IDs));
codesTree_h = cell(length(IDs),1);

fprintf('\n')
disp('Cycling through each healthy subject:')
for y = 1:length(IDs_healthy)
    subject_indices = find(IDs_healthy(y)==trainingClassifierData.subjectID);
    healthy = isolateSubject(trainingClassifierData,subject_indices);
    
    %Extract data
    features_h     = healthy.features; %features for classifier
    subjects_h     = healthy.subject;  %subject number
    uniqSubjects_h = unique(subjects_h); %list of subjects
    statesTrue_h = healthy.activity;     %all the classifier data
    subjectID_h = healthy.subjectID;
    sessionID_h = healthy.sessionID;
    uniqStates_h  = unique(statesTrue_h);
    
    n_act_h(y) = length(uniqStates_h);
    n_train_h(y) = size(features_h,1);
    
    %Generate codesTrue
    codesTrue_h = zeros(1,length(statesTrue_h));
    for i = 1:length(statesTrue_h)
        codesTrue_h(i)  = find(strcmp(statesTrue_h{i},states));
    end
    
    %Train Random Forest on Global Patient
    ntrees = 100;
    disp(['RF Train - Healthy '  num2str(IDs_healthy(y)) '  #Samples Train = ' num2str(size(features_h,1))]);
    opts_ag = statset('UseParallel',1);
    RFmodel_h = TreeBagger(ntrees,features_h,codesTrue_h,'OOBVarImp',OOBVarImp,'Options',opts_ag);
    
    %Predict with Each Tree from RF
    for p = 1:ntrees
        %Predict with each tree on fourth session
        codesTree_new = RFmodel_h.Trees{p}.predict(features_new);
        codesTree_new = str2num(cell2mat(codesTree_new));
        [~, accuracies_h(p,y)] = confusionMatrix_5(codesTrue_new,codesTree_new);
        
        %Predict with each tree on first three sessions
        codesTree_main = RFmodel_h.Trees{p}.predict(features_main);
        codesTree_h{y}(:,p) = str2num(cell2mat(codesTree_main));        
    end
end

disp('Posteriors generated for main and new sessions.')

% %% LAYER 1: SUMMARY TABLE
% layer1_tbl = table([IDs; IDs_healthy],[n_act; n_act_h],[n_train; n_train_h],[acc_new; acc_new_h],[acc_main; acc_main_h],'VariableNames',{'ID','N_Activities','Train_Size','Acc_New','Acc_Main'});
% disp(layer1_tbl);
% 
% figure;
% subplot(2,1,1)
% hold on
% plot(1:length([IDs; IDs_healthy]),[acc_new; acc_new_h],'LineWidth',2)
% plot(1:length([IDs; IDs_healthy]),[acc_main; acc_main_h],'LineWidth',2)
% ylim([0 1])
% ylabel('Accuracy','FontSize',16)
% set(gca,'Box','off','XTick',[1:(length(IDs) + length(IDs_healthy))],'XTickLabel',{},'YTick',[0.1:0.1:1],'TickDir','out','LineWidth',2,'FontSize',14,'FontWeight','bold');
% legend({'Accuracy New','Accuracy Main'},'FontSize',16)
% hold off
% 
% subplot(2,1,2)
% plot(1:length([IDs; IDs_healthy]),[n_train; n_train_h],'LineWidth',2)
% xlabel('C-Brace Subject ID','FontSize',16)
% ylabel('Training Data Size','FontSize',16)
% set(gca,'Box','off','XTick',[1:(length(IDs) + length(IDs_healthy))],'XTickLabel',num2cell([IDs; IDs_healthy]),'TickDir','out','LineWidth',2,'FontSize',14,'FontWeight','bold');

%% MAJORITY VOTING

voting_mat = zeros(length(codesTrue_main),5);

%Vote with patients
for y = 1:length(IDs)
    %Scale the codes
    for p = 1:ntrees
        voting_mat = voting_mat + codesTransform(codesTree{y}(:,p)).*exp(1/(1-accuracies(p,y)));
    end
end

%Vote with healthy
for y = 1:length(IDs_healthy)
    %Scale the codes
    for p = 1:ntrees
        voting_mat = voting_mat + codesTransform(codesTree_h{y}(:,p)).*exp(1/(1-accuracies_h(p,y)));
    end
end

%Generate predicted codes
[~, codes_FINAL] = max(voting_mat,[],2);
[matB_FINAL, acc_B_FINAL] = confusionMatrix_5(codesTrue_main,codes_FINAL);
disp(matB_FINAL)
disp(acc_B_FINAL)
