function R=mySumAbs(image1,image2)
s1=size(image1);
s2=size(image2);
image1=double(image1);
image2=double(image2);
% n parameter to control similarity metrics
% n=1 normalized cross-correlation 2sum of absolute difference  3 sum of square
% error
n=2;

if s1~=s2
    disp('size of image not match');
    return
end
%normalized cross-correlation
if n==1
   corrmatrix=image1.*image2;
   R=sum(corrmatrix);
end

if n==2
    %sum of absolute difference
    difference=image1 - image2;
    R=sum(abs(difference),"all");
end

if n==3
    %sum of square
    difference=image1 - image2;
    sqdiff=difference.^2;
    R=sum(sqdiff,"all");
end


