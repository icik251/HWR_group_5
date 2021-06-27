

%Factors for down sampling.
dsfL = 1/3 ; %for letters
dsfP = 1/1 ; %for pages

%Compute the typical SIFT features for each time period from letter sets.
X = AllSiftFeatures('characters_for_style_classification/Archaic',dsfL);
Y = AllSiftFeatures('characters_for_style_classification/Hasmonean',dsfL);
Z = AllSiftFeatures('characters_for_style_classification/Herodian',dsfL);


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



A = R./(sum(R,2)*ones(1,3)); 
FinalScores = A./(ones(6,1)*mean(A))