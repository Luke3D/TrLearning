function [cmat, acc] = confusionMatrix_5(codesTrue,codesPredicted)
    %Creates a 5x5 confusion matrix (regardless of whether there is data
    %from all 5 classes) which is important for patients that don't have
    %stars activities. Function also calculates overall accuracy. codesTrue
    %and codesPredict are numeric vectors with integer values ranging ONLY
    %from 1 to 5.
    
    if length(codesTrue) ~= length(codesPredicted)
        error('Two input vector are not the same size.')
    end
    
    cmat = zeros(5,5);
    for t = 1:length(codesTrue)
        cmat(codesTrue(t),codesPredicted(t)) = cmat(codesTrue(t),codesPredicted(t)) + 1;
    end
    acc = trace(cmat)/sum(sum(cmat));
end