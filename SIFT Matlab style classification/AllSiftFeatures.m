%Returns the Sift features of all the images in the subfolders of folderpath
function Output = AllSiftFeatures(folderpath , dsf)

%Firstly we read all images of letters and save their SIFT features.
%Find all images in subfolders with .jpg extension.
image_files = dir (strcat(folderpath,'/**/*.jpg'));
n = length(image_files); %number of images
AllSIFTFeatures = cell(n, 3); %Matrix of matrices for all SIFT features

for i=1:n
      %Check for correct directory.
      %if(~isempty(regexp(image_files(i).folder,expression, 'once')))
          path = strcat(image_files(i).folder,'\',image_files(i).name);
          picture = imresize(imread(path), dsf, 'bilinear');
          [im, des, loc] = sift(picture); %Compute SIFT features
          AllSIFTFeatures{i,1} = im;
          AllSIFTFeatures{i,2} = des;
          AllSIFTFeatures{i,3} = loc;
      %end
end
Output = AllSIFTFeatures;
