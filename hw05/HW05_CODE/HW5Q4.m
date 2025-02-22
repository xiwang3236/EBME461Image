edges=imread("edges.tif");
wheel=imread("wheel.tif");
edges=im2gray(edges);
wheel=im2gray(wheel);
edges=double(edges);
wheel=double(wheel);
[m,n]=size(edges);


edges_freq=fftshift(fft2(edges));
wheel_freq=fftshift(fft2(wheel));
ILPF=zeros(m,n);
Gau=zeros(m,n);
r=(m+n)/8;
sigma=(m+n)/16;
mu=0;

for i=1:m
    for j=1:n
        if ((m/2-i)^2+(n/2-j)^2) < r^2
        ILPF(i,j)=1;

        end
        x=sqrt((m/2-i)^2+(n/2-j)^2);
        Gau(i,j)=(1 / (sigma * sqrt(2 * pi))) * exp(-((x - mu).^2) / (2 * sigma^2));

    end
end

edges_freq_ILPF=edges_freq.*ILPF;
edges_freq_Gau=edges_freq.*Gau;
wheel_freq_ILPF=wheel_freq.*ILPF;
wheel_freq_Gau=wheel_freq.*Gau;

edges_ILPF=ifft2(ifftshift(edges_freq_ILPF));
edges_Gau=ifft2(ifftshift(edges_freq_Gau));
wheel_ILPF=ifft2(ifftshift(wheel_freq_ILPF));
wheel_Gau=ifft2(ifftshift(wheel_freq_Gau));

tiledlayout(7,2)
nexttile
imagesc(edges)
title('eages Cartesian domain')
colormap('gray')
nexttile
imagesc(wheel)
title('wheel Cartesian domain')
colormap('gray')
nexttile
imagesc(abs(edges_freq))
title('edges frequency domain')
colormap('gray')
nexttile
imagesc(abs(wheel_freq))
title('wheel frequency domain')
colormap('gray')
nexttile
imagesc(abs(edges_freq_ILPF))
title('edges ILPF frequency domain')
colormap('gray')
nexttile
imagesc(abs(wheel_freq_ILPF))
title('wheel ILPF frequency domain')
colormap('gray')
nexttile
imagesc(abs(edges_ILPF))
title('edges ILPF Cartesian domain')
colormap('gray')
nexttile
imagesc(abs(wheel_ILPF))
title('wheel ILPF Cartesian domain')
colormap('gray')
nexttile
imagesc(abs(edges_freq_Gau))
title('edges Gaussian frequency domain')
colormap('gray')
nexttile
imagesc(abs(wheel_freq_Gau))
title('wheel Gaussian frequency domain')
colormap('gray')
nexttile
imagesc(abs(edges_Gau))
title('edges Gaussian Cartesian domain')
colormap('gray')
nexttile
imagesc(abs(wheel_Gau))
title('wheel Gaussian Cartesian domain')
colormap('gray')
nexttile
imagesc(ILPF)
title('Ideal Low-pass Filter (ILPF)')
colormap('gray')
nexttile
imagesc(Gau)
title('Gaussian Low-pass Filter')
colormap('gray')