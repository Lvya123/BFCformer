clear all;
ts =0;
tp =0;
for i=1:1200                          % the number of testing samples
   x_true=im2double(imread(strcat('C:\Users\lyn\Desktop\picture\Ground truth\target-did\',sprintf('%d.jpg',i))));  % groundtruth 
   x_true = rgb2ycbcr(x_true);
   x_true = x_true(:,:,1); 
   x = im2double(imread(strcat('C:\Users\lyn\Desktop\epoch_185\',sprintf('%d_restored.png',i))));     %reconstructed image
   x = rgb2ycbcr(x);
   x = x(:,:,1);
   tp= tp+ psnr(x,x_true);
   ts= ts+ssim(x*255,x_true*255);
end
fprintf('psnr=%6f, ssim=%6f\n',tp/1200,ts/1200)



