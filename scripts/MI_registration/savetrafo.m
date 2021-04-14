% savetrafo.m
%
% Created by Elisabeth Wetzer on 20-06-09.
% Copyright © 2020 Elisabeth Wetzer. All rights reserved.
%
% Description: saves tform to file
% Input: filename, variable
% 

function savetrafo(OutputFile,tform)
% Saves transformation recovered by Registration as mat file
save(OutputFile,'tform')

end