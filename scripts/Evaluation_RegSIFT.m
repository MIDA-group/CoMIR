
% Evaluation_RegSIFT.m
%
% Created by Elisabeth Wetzer on 20-06-09.
% Copyright © 2020 Elisabeth Wetzer. All rights reserved.
%
% The script calculates the error between corner points in the reference
% patch and the corner points of the patch obtained by the registration and
% stores the error, as well as the coordinates in a .csv file


% Script to load SIFTResults.csv in which the transformations returned from
% Fiji after registration using SIFT are stored
Import_SIFTResults;

% Script to load metadata.csv in which the coordinates of the corner points
% of the reference and transformed patches are stored
ImportMetaData;

% Script to load ResultfileTemplate.csv to create a .csv file storing the
% coordinates of the corner points of the patch recovered by registration,
% the details of the modalities registered, as well as the error resulting
% from the registration
ImportResultfileTemplate;
Table=ResultfileTemplate;

for i=1:height(SIFTResults)
    
    % Get image ID out of file name, assuming that the transformations were
    % stored with the naming convention tform_ImgID_.mat for ImgID being
    % the image identifier
    tmp=char(SIFTResults.fileA(i));
    tmp2=tmp((find(tmp=='/', 1, 'last')+1):end);
    mainfilename=tmp2(3:end-10);
    
    
    % Get Transformation for recovered patch
    rotmat_rec = [SIFTResults.m00(i),SIFTResults.m01(i);SIFTResults.m10(i),SIFTResults.m11(i)];
    tx_rec=SIFTResults.m02(i);
    ty_rec=SIFTResults.m12(i);
    
    
    
    % Read out coordinates of reference patch from csv file
    RefPatch_C1=[metadata.X1_Ref(i),metadata.Y1_Ref(i)];
    RefPatch_C2=[metadata.X2_Ref(i),metadata.Y2_Ref(i)];
    RefPatch_C3=[metadata.X3_Ref(i),metadata.Y3_Ref(i)];
    RefPatch_C4=[metadata.X4_Ref(i),metadata.Y4_Ref(i)];
    
    % Read out coordinates of transformed patch from csv file
    TransPatch_C1=[metadata.X1_Trans(i),metadata.Y1_Trans(i)];
    TransPatch_C2=[metadata.X2_Trans(i),metadata.Y2_Trans(i)];
    TransPatch_C3=[metadata.X3_Trans(i),metadata.Y3_Trans(i)];
    TransPatch_C4=[metadata.X4_Trans(i),metadata.Y4_Trans(i)];
    
    % set center of rotation
    center_of_rot=[833/2,833/2];
    
    
    % Calculate coordinates of recovered patch in coordinate system of the
    % reference and transformed patch (origin in upper left corner of
    % reference patch)
    RecPatch_C1=(TransPatch_C1)*rotmat_rec-[tx_rec,ty_rec]*rotmat_rec;
    RecPatch_C2=(TransPatch_C2)*rotmat_rec-[tx_rec,ty_rec]*rotmat_rec;
    RecPatch_C3=(TransPatch_C3)*rotmat_rec-[tx_rec,ty_rec]*rotmat_rec;
    RecPatch_C4=(TransPatch_C4)*rotmat_rec-[tx_rec,ty_rec]*rotmat_rec;
    
    
    % Calculate error between corners of reference patch and recovered
    % patch
    error=[sqrt((RefPatch_C1(1)-RecPatch_C1(1))^2+(RefPatch_C1(2)-RecPatch_C1(2))^2),...
        sqrt((RefPatch_C2(1)-RecPatch_C2(1))^2+(RefPatch_C2(2)-RecPatch_C2(2))^2),...
        sqrt((RefPatch_C3(1)-RecPatch_C3(1))^2+(RefPatch_C3(2)-RecPatch_C3(2))^2),...
        sqrt((RefPatch_C4(1)-RecPatch_C4(1))^2+(RefPatch_C4(2)-RecPatch_C4(2))^2)]
    
    
    % Fill in Results Table
    % Fill in which registration method was used (alpha-AMD, SIFT, MI, etc)
    Table.Method(i)='SIFT';
    
    
    if SIFTResults.success(i)==1
        
        % Mean error between reference corner points and corner points of patch
        % recovered by registration
        Table.Error(i)=mean(error);
        
        % Coordinates of recovered patch in the same coordinate system as the
        % corner points of the registered and transformed patch (origin upper
        % left corner of reference patch)
        Table.X1_Recover(i)=RecPatch_C1(1);
        Table.Y1_Recover(i)=RecPatch_C1(2);
        
        Table.X2_Recover(i)=RecPatch_C2(1);
        Table.Y2_Recover(i)=RecPatch_C2(2);
        
        Table.X3_Recover(i)=RecPatch_C3(1);
        Table.Y3_Recover(i)=RecPatch_C3(2);
        
        Table.X4_Recover(i)=RecPatch_C4(1);
        Table.Y4_Recover(i)=RecPatch_C4(2);
        
        % Registrations larger than 100 are considered to be a failure, so set all
        % errors to 101 if no corresponding feature points were found to allow for
        % SIFT registration
    elseif SIFTResults.success(i)==0
        Table.Error(i)=101;
        
    end
    
    
    % Fill in which modality was used as a reference image
    Table.ReferenceImage(i)='BF';
    % Fill in which modality was used as a moving/floating image
    Table.FloatingImage(i)='SHG';
    
    
end

% Write table to results file in csv format
writetable(Table, 'SIFT_Result.csv')
