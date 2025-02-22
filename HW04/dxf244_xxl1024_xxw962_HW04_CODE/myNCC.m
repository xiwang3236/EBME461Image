function R=myNCC(image1,image2)
s1=size(image1);
s2=size(image2);
% n parameter to control similarity metrics
% n=1 normalized cross-correlation 2sum of absolute difference  3 sum of square
% error
n=1;

if s1~=s2
    disp('size of image not match');
    return
end
%normalized cross-correlation
if n==1
   bar_fl=mean(image1,"all");
   bar_drr=mean(image2,'all');
   Dfl=image1-bar_fl;
   Ddrr=image2-bar_drr;
   corr=Dfl.*Ddrr;
   a=sum(corr,"all");
   c=sum(Dfl.^2,"all");
   d=sum(Ddrr.^2,'all');

   R=a/sqrt(c*d);
end

end


