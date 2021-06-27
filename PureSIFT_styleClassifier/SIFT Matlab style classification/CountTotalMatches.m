function num = CountTotalMatches( LettersSift , page , dsf )

total = 0; %total number of matches
n = size(LettersSift,1);

archaic821 = imresize( imread(page), dsf, 'bilinear');

nx = size(archaic821,1);
ny = size(archaic821,2);
G = 2000;
ix = [1:G:nx,nx];
iy = [1:G:ny,ny];
for kx = 1:(size(ix,2)-1) %split image horizontally
    for ky = 1:(size(iy,2)-1) %split image vertically
        [im1, des1, loc1] = sift(archaic821((ix(kx)):(ix(kx+1)),(iy(ky)):(iy(ky+1))));
        for i = 1 : n
            im2 = LettersSift{i,1};
            des2 = LettersSift{i,2};
            loc2 = LettersSift{i,3};
            if (size(des1,1)>2) && (size(des2,1)>2) %if there are any sift features
            total = total + match2(im1, des1, loc1, im2, des2, loc2);
            end
        end
    end
end
num = total;
end