function [codesTransformed] = codesTransform(codes)
    %Transforms a nx1 vector of codes into a nx5 vector with binary values

    codesTransformed = zeros(length(codes),5);
    
    for z = 1:length(codes)
        codesTransformed(z,codes(z)) = 1;
    end
end