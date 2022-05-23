%%  Final Year Project
%   Offside Detection Code
%   Classifier algorithm to detect offside player 
%   by Khalid Malami
%   23/05/2022

%% Initialisation
clear
close all
clc

%% Read Video
obj = VideoReader('80.mp4'); % create video object
start_frame = 1;            % start frame
offside_frame = 3;          % frame at which the pass is made

for frame_index = 1:4       % parsing through video frames
    if(exist('img'))
        prev_img = img;
    end
    img = readFrame(obj);
    
    if(frame_index<start_frame)
        continue
    end
    
%% Getting Vanishing Point lines
    % Requesting parallel vanishing lines input from user
    if (frame_index == start_frame)
        imshow(img)
        [x,y] = getpts;
        points = [x,y];
        close all
        tStart = cputime;
    % Calculating Slope
    m = zeros(size(points,1)/2,1);          % slope matrix   
    c = zeros(size(points,1)/2,1);          % intercept matrix             
    k = 1;
    vp = zeros(2,1);                        % vanishing point matrix
    for j = 1:2:size(points,1)
        m(k) = (points(j+1,2) - points(j,2)) / (points(j+1,1) - points(j,1));
        c(k) = - points(j,1) * m(k) + points(j,2);
        k = k+1;
    end
    % Calculating the Vanishing point
    count = 0;
    for p = 1:size(points,1)/2
       for q = p+1:size(points,1)/2
           count = count + 1;
           A = [-m(p),1;-m(q),1];
           b = [c(p);c(q)];
           vp = vp + A\b;
       end
    end
    vp = int16(vp/count);
    end
%% Actual Detection starts (one every 20 frames).
    BW_img = rgb2gray(img);                 % Coverting the image to grayscale
    Edge_img = edge(BW_img,'sobel');        % Converting greyscale image to edge image using Sobel
    
% Removing the TOP Boundary using Hough Transform
%   Defining Hough Parameters
    start_angle = 89;                       
    end_angle = 89.99;
    theta_resolution = 0.01;
%   Obtaining Hough coefficients    
    [hou,theta,rho] = hough(Edge_img(1:floor(size(Edge_img,1)/2),:), 'Theta', start_angle:theta_resolution:end_angle);
    peaks = houghpeaks(hou,2,'threshold',ceil(0.3*max(hou(:))));
    lines = houghlines(Edge_img(1:floor(size(Edge_img,1)/2),:),theta,rho,peaks,'FillGap',5,'MinLength',7);
%   Identifying longest horizontal lines    
    min_row = lines(1).point1(2);
    xy_long = [lines(1).point1; lines(1).point2];
    
    for k = 1:length(lines) 
        xy = [lines(k).point1; lines(k).point2];
        row_index = lines(k).point1(2);
        if (row_index < min_row)
            min_row = row_index; 
            xy_long = xy;
            index = k;
        end
    end
    
%   Removing top boundary pixels
    img(1:xy_long(:,2),:,:)=0;
    BW_img(1:xy_long(:,2),:,:)=0;
    Edge_img(1:xy_long(:,2),:,:)=0;
%    
%% Determining the actual play area
%   Find all dominant greens
    indg = find(fuzzycolor(im2double(img),'green')<0.1);
    n = size(img,1)*size(img,2);
%   Image Processing to find connected components
    imggreen = img;
    imggreen([indg;indg+n;indg+2*n]) = 0;
    
    mask = imbinarize(rgb2gray(imggreen));
    mask = imfill(mask,'holes'); 
    mask_open = bwareaopen(mask,300);
    mask_open = imfill(mask_open,'holes'); 
    Conn_Comp_green = bwconncomp(mask_open,8);
    S_green = regionprops(Conn_Comp_green,'BoundingBox','Area');
%   Largest connected component
    [~,max_ind_green] = max([S_green.Area]);
    bb_max_green = S_green(max_ind_green).BoundingBox;
    
    %Get a new `valid' image, which contatins only the actual play area.
    img_valid = img;
    max_h = size(img,1);
    if(bb_max_green(1)>1)
        for row = 1:max_h
            x_curr  = ((bb_max_green(1)-vp(1))/(max_h-vp(2))) * (row-max_h);
            x_curr = floor(x_curr);
            img_valid(row,1:x_curr,:) = 0;
        end
    else
        current_ind = 1;
        while(1)
            current_value = mask_open(current_ind,1);
            if(current_value ==1)
                break
            end
            current_ind = current_ind + 1;
        end
        for row = 1:current_ind
            x_curr  = bb_max_green(1) + ((bb_max_green(1) -vp(1))/(current_ind-vp(2))) * (row-current_ind);
            x_curr = floor(x_curr);
            img_valid(row,1:x_curr,:) = 0;
        end

    end
    
%% Determining the players and Team_Ids
    
%   Define attacking team colours
    indg = find(fuzzycolor(im2double(img_valid),'red')<0.1);
    n = size(img,1)*size(img,2);

    img_team_read = img_valid;
    img_team_read([indg;indg+n;indg+2*n]) = 0;
    
%   Image processing
    mask = imbinarize(rgb2gray(img_team_read));
    mask = imfill(mask,'holes'); 
    mask_open = bwareaopen(mask,30);
    mask_open = imfill(mask_open,'holes');
    S_E_D = strel('disk',15);
    mask_open = imdilate(mask_open,S_E_D);
    Conn_Comp_team_red = bwconncomp(mask_open,8);
    S_team_red = regionprops(Conn_Comp_team_red,'BoundingBox','Area');
    
%   Define defending team colours
    indg = find(fuzzycolor(im2double(img_valid),'blue')<0.1);
    n = size(img,1)*size(img,2);
    img_team_read = img_valid;
    img_team_read([indg;indg+n;indg+2*n]) = 0;
    mask = imbinarize(rgb2gray(img_team_read));
    mask = imfill(mask,'holes');
    mask_open = bwareaopen(mask,15);
    mask_open = imfill(mask_open,'holes'); 
    S_E_D = strel('disk',15);
    mask_open = imdilate(mask_open,S_E_D);
    Conn_Comp_team_blue = bwconncomp(mask_open,8);
    S_team_blue = regionprops(Conn_Comp_team_blue,'BoundingBox','Area');

%   Getting all players/teamids in one list
    S = [S_team_red; S_team_blue];
    Team_Ids = [ones(size(S_team_red,1),1); 2*ones(size(S_team_blue,1),1)];
    Players = cat(2,[vertcat(S(1:size(S,1)).BoundingBox)], Team_Ids);
%% Mark the bounding boxes
    f = figure('visible','off');    
    left_most = 9999;
    team_index = 5;
    [~,last_player] = min(Players(:,1));
    
% Detecting if furthest attacking player is offside
    if(frame_index == offside_frame && Players(last_player, team_index)==2)
        disp("Offside detected")
        figure();
        imshow(img)
        hold on;
    elseif(frame_index == offside_frame && Players(last_player, team_index)==1)
        disp("Failed to Detect Offside")
        figure();
        imshow(img)
        hold on;
    end
    
    for i =1:size(S,1)
        BB = S(i).BoundingBox;
        if(Team_Ids(i)==1)
            text(BB(1)-2, BB(2)-2,'D');
            BB(4)  = 1.5*BB(4);
            S(i).BoundingBox(4) = BB(4);
        end
        if(Team_Ids(i)==2)
            text(BB(1)-2, BB(2)-2,'A');
        end
        rectangle('Position',[BB(1),BB(2),BB(3),BB(4)],...
        'LineWidth',2,'EdgeColor','red')

        x1 = floor(BB(1)+BB(3)/4);
        y1 = floor(BB(2) + BB(4));
        ly = size(img,1);
        slope = (vp(2) - y1)/(vp(1) - x1);
        y_int = - x1 * slope + y1;
        lx = (ly - y_int)/slope;
        if(lx<left_most && Team_Ids(i) == 1)
         left_most = lx;
        end         
    end
%   Plot offside Line
    plot([left_most,vp(1)],[ly ,vp(2)],'c','LineWidth',1)
end
tEnd = cputime - tStart
