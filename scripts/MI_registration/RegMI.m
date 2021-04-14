% RegMI.m
%
% Created by Elisabeth Wetzer on 20-06-09.
% Copyright © 2020 Elisabeth Wetzer. All rights reserved.
%
% The script registers multimodal or monomodal image pairs given in the
% directories FilesRef and FilesFloat by Mattes Mutual Information using a
% (1+1) Evolutionary Optimizer. Images in the folder FilesRef and
% FilesFloat are expected to be named Img1 = R_ImgID_mod1.tif and 
% Img2 = T_ImgID_mod2.tif for R the reference image in modality 1, T the
% transformed image in modality 2 and ImgID the identifier of the image
% pair.
% Note that both the reference and moving/floating image need to be
% 1-channel.



% Path to reference (fixed) images
FilesRef=dir('TestSet/R*.tif');
% Path to transformed (moving, floating) images
FilesFloat=dir('TestSet/T*.tif');
% Path to folder in which to store transformations obtained
ResultsPath='TestSet/Results';

if ~exist(ResultsPath, 'dir')
    mkdir(ResultsPath)
end

% Choose 'multimodal' to choose MI as a metric
[optimizer, metric] = imregconfig('multimodal');

% Set hyperparameters for the optimizer and for the mutual information
optimizer = registration.optimizer.OnePlusOneEvolutionary;
optimizer.InitialRadius = 0.00001;
optimizer.Epsilon = 1.5e-8;
optimizer.GrowthFactor = 1.0001;
optimizer.MaximumIterations = 1500;
registration.metric.MattesMutualInformation.NumberOfHistogramBins=80;

% uncomment next line to parallelize the loop
%parfor i=1:length(FilesFloat)
for i=1:length(FilesFloat)
    
    
    % Load fixed image
    filenameFixed=[FilesRef(i).folder,'/',FilesRef(i).name];
    Fixed = imread(filenameFixed);
    
    
    % Load moving/floating image
    filenameMoving=[FilesFloat(i).folder,'/',FilesFloat(i).name];
    Moving = imread(filenameMoving);
    
    
    % Registration using Mattes MI
    tform = imregtform(Moving,Fixed,'rigid',optimizer,metric);
    
    % Create name for output file name. Transformation will be saved as
    % .mat file
    tmp=FilesFloat(i).name;
    str=tmp(3:end-7);
    OutputFile=[ResultsPath,'/tform_',str,'.mat'];
    
    % Transformation will be saved as .mat file
    savetrafo(OutputFile,tform)
end


