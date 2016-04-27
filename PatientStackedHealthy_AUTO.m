%% DIRECTORIES
clear all, close all;

subjects = [1 2];
repetitions = 1;
patient_stairs = [2 8 11 12 14 15];
disp(patient_stairs);
accuracy_h = zeros(repetitions,length(subjects));

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


%% CYCLE THROUGH ALL PATIENTS
for w = 1:length(subjects)
    clearvars -except subjects repetitions w patient_stairs accuracy_h p currentDir OOBVarImp
    fprintf('\n')
    disp(['PATIENT: ' num2str(subjects(w))])
    
    for m = 1:repetitions
        %% LOAD DATA TO ANALYZE
        population = 'patient';
        filename = ['trainData_' population '.mat'];
        load(filename)
        
        %% SELECT PATIENT
        all_subjectID = trainingClassifierData.subjectID;
        
        subject_analyze = subjects(w);
        subject_indices = find(subject_analyze==all_subjectID);
        cData_temp2 = isolateSubject(trainingClassifierData,subject_indices);
        
        for zz = 1:length(cData_temp2.subject)
            temp = char(cData_temp2.subject(zz));
            cData_temp2.subjectBrace(zz) = {temp(7:9)};
        end
        brace_analyze = 'Cbr';
        cData_temp = isolateBrace(cData_temp2,brace_analyze);
        
        max_sessions = 3;
        min_sessions = 1;
        cData = isolateSession(cData_temp,max_sessions,min_sessions);
        
        fprintf('\n')
        disp('These are the subjects that will be analyzed: ')
        disp(unique(cData.subject))
        fprintf('\n')
        
        disp(['REPETITION: ' num2str(m)])
        disp('Initiating Layer 1...')
        disp('Cycling through each global patient:')
        
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
        posteriors_new = []; %stores all posteriors to be generated
        posteriors_main = []; %stores all posteriors to be generated
        n_act = zeros(length(IDs),1);
        n_train = zeros(length(IDs),1);
        acc_new = zeros(length(IDs),1);
        acc_main = zeros(length(IDs),1);
        acc_act_new = cell(length(IDs),1); %stores acccuracies for each activity for each patient
        
        disp('Initiating Layer 1...')
        disp('Cycling through each global patient:')
        %Cycle through each patient
        for y = 1:length(IDs)
            
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
            ntrees = 100;
            disp(['RF Train - Patient '  num2str(IDs(y)) '  #Samples Train = ' num2str(size(features_p,1))]);
            opts_ag = statset('UseParallel',1);
            RFmodel_p = TreeBagger(ntrees,features_p,codesTrue_p,'OOBVarImp',OOBVarImp,'Options',opts_ag);
            
            %Test Random Forest on Main Sessions
            [codesRF_main,P_RF_main] = predict(RFmodel_p,features_main);
            codesRF_main = str2num(cell2mat(codesRF_main));
            
            %Test Random Forest on New Session6
            [codesRF_new,P_RF_new] = predict(RFmodel_p,features_new);
            codesRF_new = str2num(cell2mat(codesRF_new));
            
            %Collect Accuracies
            [matRF_main, acc_main(y)] = confusionMatrix_5(codesTrue_main,codesRF_main);
            [matRF_new, acc_new(y)] = confusionMatrix_5(codesTrue_new,codesRF_new);
            
            %Collect Posteriors
            posteriors_main = [posteriors_main P_RF_main];
            posteriors_new = [posteriors_new P_RF_new];
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
        acc_new_h = zeros(length(IDs_healthy),1);
        acc_main_h = zeros(length(IDs_healthy),1);
        
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
            RFmodel_p = TreeBagger(ntrees,features_h,codesTrue_h,'OOBVarImp',OOBVarImp,'Options',opts_ag);
            
            %Test Random Forest on Main Sessions
            [codesRF_main,P_RF_main] = predict(RFmodel_p,features_main);
            codesRF_main = str2num(cell2mat(codesRF_main));
            
            %Test Random Forest on New Session6
            [codesRF_new,P_RF_new] = predict(RFmodel_p,features_new);
            codesRF_new = str2num(cell2mat(codesRF_new));
            
            %Collect Accuracies
            [matRF_main, acc_main_h(y)] = confusionMatrix_5(codesTrue_main,codesRF_main);
            [matRF_new, acc_new_h(y)] = confusionMatrix_5(codesTrue_new,codesRF_new);
            
            %Collect Posteriors
            posteriors_main = [posteriors_main P_RF_main];
            posteriors_new = [posteriors_new P_RF_new];
        end
        
        disp('Posteriors generated for main and new sessions.')
        
        %% LAYER 1: GENERATE FEATURES
        features_main = getFeaturesTR(posteriors_main);
        features_new = getFeaturesTR(posteriors_new);
        disp('Feature generation complete')
        disp('Layer 1 complete.')
        fprintf('\n')
        
        %% LAYER 2: TRAIN CLASSIFIER + GENERATE FINAL ACTIVITY PREDICTIONS
        disp('Initiating Layer 2...')
        
        %Random Forest (RF)
        disp('Random Forest (RF):')
        ntrees = 100;
        RFmodel = TreeBagger(ntrees,features_new,codesTrue_new,'OOBVarImp',OOBVarImp,'Options',opts_ag);
        [codesRF_FINAL,P_RF_FINAL] = predict(RFmodel,features_main);
        codesRF_FINAL = str2num(cell2mat(codesRF_FINAL));
        [matRF_FINAL, acc_RF_FINAL] = confusionMatrix_5(codesTrue_main,codesRF_FINAL);
        disp(matRF_FINAL)
        disp(acc_RF_FINAL)
        
        %Logistic Regression (LR)
        % disp('Logistic Regression (LR):')
        % B = mnrfit(features_new,codesTrue_new);
        % pihat = mnrval(B,features_main);
        % [~, codesLR_FINAL] = max(pihat,[],2);
        % [matLR_FINAL,acc_LR_FINAL,~] = createConfusionMatrix(codesTrue_main,codesLR_FINAL);
        % disp(matLR_FINAL)
        % disp(acc_LR_FINAL)
        
        %Linear Support Vector Machine (SVM)
        % disp('Linear Support Vector Machine (SVM):')
        % options = statset('UseParallel',1);
        % template = templateSVM('KernelFunction', 'linear', 'PolynomialOrder', [], 'KernelScale', 'auto', 'BoxConstraint', 1, 'Standardize', 1);
        % trainedClassifier = fitcecoc(features_new, codesTrue_new, 'Learners', template, 'FitPosterior', 1, 'Coding', 'onevsone', 'ResponseName', 'outcome','Options',options);
        % [codesSVM_FINAL, ~, ~, ~] = predict(trainedClassifier,features_main);
        % [matSVM_FINAL,acc_SVM_FINAL,~] = createConfusionMatrix(codesTrue_main,codesSVM_FINAL);
        % disp(matSVM_FINAL)
        % disp(acc_SVM_FINAL)
        
        disp('Layer 2 complete.')
        
        accuracy_h(m,w) = acc_RF_FINAL;
    end
    
    clear cData_temp2 cData
end

save('accurach_h.mat','accuracy_h')

%% PLOT FINAL RESULTS
%Box and grouped bar plots
figure;
subplot(2,1,1)
boxplot(accuracy_h)

subplot(2,1,2)
bar(accuracy_h')

%Bar: Mean +/- SEM
%figure;