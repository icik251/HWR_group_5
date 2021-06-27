function StyleClassifier(TestImageFolderPath)

image_files = dir(strcat(TestImageFolderPath,'/*.jpg')); %get all test image paths
n = length(image_files); %number of images

%Factors for down sampling.
dsfL = 1/3 ; %for letters
dsfP = 1/1 ; %for pages

%Compute the typical SIFT features for each time period from letter sets.
X = AllSiftFeatures('characters_for_style_classification/Archaic',dsfL);
Y = AllSiftFeatures('characters_for_style_classification/Hasmonean',dsfL);
Z = AllSiftFeatures('characters_for_style_classification/Herodian',dsfL);


%TRAIN:
%Count Matches of SIFT for all combinations of pages and time periods.

R(1,1) = CountTotalMatches( X , 'full_images_periods/Archaic/archaic-821.jpg' ,dsfP);
R(1,2) = CountTotalMatches( Y , 'full_images_periods/Archaic/archaic-821.jpg' ,dsfP);
R(1,3) = CountTotalMatches( Z , 'full_images_periods/Archaic/archaic-821.jpg' ,dsfP);
R(2,1) = CountTotalMatches( X , 'full_images_periods/Archaic/archaic-1110.jpg' ,dsfP);
R(2,2) = CountTotalMatches( Y , 'full_images_periods/Archaic/archaic-1110.jpg' ,dsfP);
R(2,3) = CountTotalMatches( Z , 'full_images_periods/Archaic/archaic-1110.jpg' ,dsfP);

R(3,1) = CountTotalMatches( X , 'full_images_periods/Hasmonean/hasmonean-330-1.jpg' ,dsfP);
R(3,2) = CountTotalMatches( Y , 'full_images_periods/Hasmonean/hasmonean-330-1.jpg' ,dsfP);
R(3,3) = CountTotalMatches( Z , 'full_images_periods/Hasmonean/hasmonean-330-1.jpg' ,dsfP);
R(4,1) = CountTotalMatches( X , 'full_images_periods/Hasmonean/hasmonean-674.jpg' ,dsfP);
R(4,2) = CountTotalMatches( Y , 'full_images_periods/Hasmonean/hasmonean-674.jpg' ,dsfP);
R(4,3) = CountTotalMatches( Z , 'full_images_periods/Hasmonean/hasmonean-674.jpg' ,dsfP);

R(5,1) = CountTotalMatches( X , 'full_images_periods/Herodian/herodian-582.jpg' ,dsfP);
R(5,2) = CountTotalMatches( Y , 'full_images_periods/Herodian/herodian-582.jpg' ,dsfP);
R(5,3) = CountTotalMatches( Z , 'full_images_periods/Herodian/herodian-582.jpg' ,dsfP);
R(6,1) = CountTotalMatches( X , 'full_images_periods/Herodian/herodian-608.jpg' ,dsfP);
R(6,2) = CountTotalMatches( Y , 'full_images_periods/Herodian/herodian-608.jpg' ,dsfP);
R(6,3) = CountTotalMatches( Z , 'full_images_periods/Herodian/herodian-608.jpg' ,dsfP);


%Obtain log ratios:
S(1,3) = (log(R(1,1))-log(R(1,2))+log(R(2,1))-log(R(2,2)))/2;
S(1,1) = (log(R(1,2))-log(R(1,3))+log(R(2,2))-log(R(2,3)))/2;
S(1,2) = (log(R(1,3))-log(R(1,1))+log(R(2,3))-log(R(2,1)))/2;

S(2,3) = (log(R(3,1))-log(R(3,2))+log(R(4,1))-log(R(4,2)))/2;
S(2,1) = (log(R(3,2))-log(R(3,3))+log(R(4,2))-log(R(4,3)))/2;
S(2,2) = (log(R(3,3))-log(R(3,1))+log(R(4,3))-log(R(4,1)))/2;

S(3,3) = (log(R(5,1))-log(R(5,2))+log(R(6,1))-log(R(6,2)))/2;
S(3,1) = (log(R(5,2))-log(R(5,3))+log(R(6,2))-log(R(6,3)))/2;
S(3,2) = (log(R(5,3))-log(R(5,1))+log(R(6,3))-log(R(6,1)))/2;


%TEST:

%make results folder
mkdir 'results'


%Classification:
for i=1:n
    path = strcat(image_files(i).folder,'\',image_files(i).name);
    %Count the number of matches:
    r1 = CountTotalMatches( X , path ,dsfP);
    r2 = CountTotalMatches( Y , path ,dsfP);
    r3 = CountTotalMatches( Z , path ,dsfP);
    s(1,3) = log(r1/r2);
    s(1,1) = log(r2/r3);
    s(1,2) = log(r3/r1);
    
    Q = ones(3,1)*s(1,1:3) - S;
    q = sum(Q.*Q,2);
    
    %write the txt file
    name = image_files(i).name;
    name((size(name,2)-3):(size(name,2)+6)) = '_style.txt';
    fid = fopen( strcat('results/', name) , 'wt' );
    
    if(min(q)==q(1,1))
        fprintf( fid, "Archaic");end
    if(min(q)==q(2,1))
        fprintf( fid, "Hasmonean");end
    if(min(q)==q(3,1))
        fprintf( fid, "Herodian");end
    fclose(fid);                
    
end


