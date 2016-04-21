clear all
load trainData_healthy.mat
indWalk = strcmp(trainingClassifierData.activity,'Walking');
X_Walk = trainingClassifierData.features(indWalk);  
Y_Walk = trainingClassifierData.subjectID(indWalk); %label correspond to subject
mappedX = tsne(X_Walk,Y_Walk);
figure
h = gscatter(mappedX(:,1),mappedX(:,2),Y_Walk,[],'o',6)