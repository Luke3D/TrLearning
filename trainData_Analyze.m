%% LOAD TRAINDATA.MAT FILE
cd(fileparts(which('trainData_Analyze.m')))
currentDir = pwd;
addpath([pwd '/sub']); %create path to helper scripts

%trainData.mat file to analyze
population = 'patient';
filename = ['trainData_' population '.mat'];
load(filename)

%% GENERATE PROFILE
data = trainingClassifierData;
clear trainingClassifierData

IDs = unique(data.subjectID);
ID_ind = data.subjectID;
states = {'Sitting';'Stairs Dw';'Stairs Up';'Standing';'Walking'};

%Go through each subject
for a = 1:length(IDs)
    %Isolate subject
    subject_ind = find(IDs(a)==ID_ind);
    subject = isolateSubject(data,subject_ind);
    
    %Convert subject ID to string
    if IDs(a) < 10
        subj_str = ['0' num2str(IDs(a))];
    elseif IDs(a) > 9
        subj_str = num2str(IDs(a));
    end
    
    %Determine number of braces for subject
    for zz = 1:length(subject.subject)
        temp = char(subject.subject(zz));
        subject.subjectBrace(zz) = {temp(7:9)};
    end
    braces = unique(subject.subjectBrace);
    N_brace = length(unique(subject.subjectBrace)); %number of braces
    
    disp(['SUBJECT: CBR' subj_str])
    
    %Go through each brace
    for b = 1:N_brace
        brace = isolateBrace(subject,braces(b));        
        
        disp([braces{b} ':'])
        
        %Determine number of sessions
        sessionIDs = unique(brace.sessionID);
        N_sessions = length(sessionIDs); %number of sessions
        
        %Go through each session
        for c = 1:N_sessions
            session = isolateSession(brace,c,c); %isolate sesions
            
            %Generate codesTrue
            codesTrue = zeros(1,length(session.features));
            for i = 1:length(session.features)
                codesTrue(i) = find(strcmp(session.activity{i},states));
            end
            
            disp(['   Session ' num2str(sessionIDs(c))])
            
            %Output amount of data
            for d = 1:length(states)
                N_act = length(find(codesTrue == d));
                pct = N_act./length(codesTrue)*100;
                disp(['      ' states{d} ': ' num2str(N_act) ' (' num2str(pct) '%)'])
            end
        end
    end
    fprintf('\n')
end