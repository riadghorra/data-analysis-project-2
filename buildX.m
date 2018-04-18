% Build X and y
%{
Uses the feature extraction functions and runs through all the folders
defined in "places". The original path browser should be pointed to the top
folder containing the "places", with folders named similary to the places.

The script then runs through all the folders and extracts the features for
each. It is expected that a fog_index_txt files is present in each image
folder, that has the labelling of the images.

One can add more features in the X =... by including them in the vector.

If including feat_expEdge feature, a *places*.mat file needs to be present.
This .mat file can be made with the buildRefEdgeIm.m script
%}

path_to_im = uigetdir('C:\Users\shahab\Desktop\DTU\2018S\02582 Computational Data Analysis\Case 2\Case 2\online_rep'); %choose the parent folder of images's when you run it

places = {'\images\'};

% Whether to normalize or not
normalize = 1;

X = [];
y = [];

im_abs_path = {};

for i = 1:length(places)
    % Places is different cameras
    path_to_ims = [path_to_im, places{i}];
    dirs_im = dir(path_to_ims);
    

    % Loop through all date folders in place folder
    for n = 3:length(dirs_im)
        
        dirs_ims_in_dir = dir([path_to_ims, dirs_im(n).name]);
        filename = [path_to_ims, dirs_im(n).name,'\', 'fog_index.txt'];
        delimiter = ',';
        startRow = 2;

        formatSpec = '%s%f%[^\n\r]';

        fileID = fopen(filename,'r');

        dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines' ,startRow-1, 'ReturnOnError', false);

        fclose(fileID);

        names_fog = dataArray{:, 1};
        index_fog = dataArray{:, 2};

        clearvars filename delimiter startRow formatSpec fileID ans;

        empty = [];
        
%         load([path_to_im places{i}(1:end-1) '.mat'])
        
        for k = 3:length(dirs_ims_in_dir)
            % Abs path of image in question
            
            % Extract features for each im and append
            absolut_file = [path_to_ims, dirs_im(n).name,'\', dirs_ims_in_dir(k).name];
            file_desc = dir(absolut_file);
            if file_desc.bytes > 100 
                % Some images is empty, or not images
                if strcmp(file_desc.name, 'fog_index.txt') == 0
                    im_temp = imread(absolut_file);

                    im_temp_double = im2double(rgb2gray(im_temp));

                    
                    % All feature calculations added here, which is
                    % calculated for each individual image.
                    
                    [var_lap, ~] = feat_varlaplacian(im_temp_double);
                    [sum_lap, ~] = feat_sumlaplacian(im_temp_double);                    
                    
                    [dark_feat, ~] = feat_dark(im_temp, 9);
                    
                    %[feat_locr, ~] = feat_locvar(im_temp_double, 9);
                    
                    [feat_varsob, ~] = feat_varsobel(im_temp_double);
                    [feat_tensob, ~] = feat_tensobel(im_temp_double);
                    
%                     [feat_expEdge, ~] = feat_expectedEdges(ref, im_temp_double);
                    
                    [feat_dark_cent, ~] = feat_dark_center(im_temp_double, 9);
                    
                    [feat_dark_light, ~] = feat_light_dark_diff(im_temp, 9);
                    
                    feat_BurnPct = feat_burn(im_temp);
                                        
                    % Add feature to the feature matrix X
                    %
                    X = [X ; [var_lap, sum_lap, dark_feat, feat_varsob, feat_tensob, feat_dark_cent, feat_dark_light, feat_BurnPct]];
                    
                    im_abs_path = [im_abs_path; [path_to_ims, dirs_im(n).name,'\', dirs_ims_in_dir(k).name]];
                end
                
            else
                % If image is empty, notice placement such that it can be
                % removed from fog index
                empty = [empty; k];
            end
            
            if(mod(k,200) == 0)
                disp(k / 4781 * 100);
            end

        end
        
        % Make y at end (so we can remove fields with empty image
        index_fog(empty) = empty;

        
        
        y = [y; index_fog(1:end)];
        
        disp(['Length of y: ', num2str(length(y))]);
        disp(['Length of X: ', num2str(length(X(:,1)))]);
        disp(' ');
        
        if length(y) ~= length(X(:,1))
            disp('length drift');
            return;
        end
    end
    
    % Done with one place, index placement of shift
    camera_coding(i) = length(X(:,1));
end

if normalize == 1
    % Camera change at 561
    X_norm = zeros(size(X));
    camera_coding = [561, camera_coding];
    
    for i=1:length(camera_coding)
        if i == 1
            for k=1:length(X(1,:))
                [X_norm(1:camera_coding(i), k),m,sta] = std_norm(X(1:camera_coding(i), k));
                %X(1:camera_coding(i),k) = norm(X(1:camera_conding(i), k));
            end
        else
            for k = 1:length(X(1,:))
                [ X_norm(camera_coding(i-1)+1:camera_coding(i),k),m,sta] = std_norm( X(camera_coding(i-1)+1:camera_coding(i),k));
                
               % X(camera_coding(i-1):camera_coding(i),k) = norm(X(camera_coding(i-1):camera_coding(i),k));
            end
        end
    end
end

attributeNames = {'variance laplacian', 'Sum laplacian','Dark Channel','Variance Sobel', 'Ten Sobel', 'Expected edge', 'Dark channel center', 'Dark Light channel', 'Burn Pct'};
y_fog = find(y== 1);
