function [out_mat] = interpolate_matrix(in_mat,Year_length,interval_length,set1)
% Interpolates values to the given length with different setting values
% set1=1 is for linear interpolation between points
% set1=2 is for repeating value between points
    if set1 == 1     
        [m01,n01] = size(in_mat);
        if (m01 ~= Year_length*interval_length)     % Interpolate to expand matrix if too short
            if mod(Year_length*interval_length,m01)>0, error('Cannot interpolate matrix: mismatched dimensions'), end
            out_mat = zeros(m01*interval_length,n01);
            for i1=1:m01-1
                for i2=1:interval_length
                    out_mat((i1-1)*interval_length+i2,:) = (in_mat(i1+1,:)-in_mat(i1,:))/4*(i2-1)+in_mat(i1,:);
                end
            end
            out_mat((m01-1)*interval_length+1:m01*interval_length,:) = repmat(in_mat(m01,:),interval_length,1);
        else
            out_mat = in_mat;
        end
    elseif set1 == 2 
        [m01,n01] = size(in_mat);
        if (m01 ~= Year_length*interval_length)     % Interpolate to expand matrix if too short
            out_mat = zeros(m01*interval_length,n01);
            out_mat = reshape(permute(repmat(in_mat,1,1,interval_length),[3,1,2]),[],n01);
        else
            out_mat = in_mat;
        end
    end
end

