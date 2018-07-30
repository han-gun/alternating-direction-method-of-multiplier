%%  REFERENCE
% https://en.wikipedia.org/wiki/Augmented_Lagrangian_method#cite_note-5
% ftp://ftp.math.ucla.edu/pub/camreport/cam08-29.pdf

%% COST FUNCTION
% x^* = argmin_x { 1/2 * || A(X) - Y ||_2^2 + lambda/2 * ( || D_x(X) ||_2^2 + || D_y(X) ||_2^2 ) }
%
%% Augmented Lagrangian Method when s.t. Z = X
% argmin_X { 1/2 * || A(X) - Y ||_2^2 + lambda/2 * ( || Dx ||_2^2 + || Dy ||_2^2 ) }
% s.t. Dx = D_x(X)
%      Dy = D_y(X)
%
% By augmented lagrangian method,
% L(X, Dx, Dy, Bx, By) = 1/2 * || A(X) - Y ||_2^2 + lambda/2 * ( || Dx ||_2^2 + || Dy ||_2^2 ) + rho/2 * ( || Dx - D_x(X) - Bx ||_2^2 + || Dy - D_y(X) - By ||_2^2 )
%
% SubProb. 1 : argmin_X = L(X, Dx^k, Dy^k, Bx^k, By^k)
% X^k+1 = argmin_X { 1/2 * || A(X) - Y ||_2^2 + rho/2 * ( || Dx^k - D_x(X) - Bx^k ||_2^2 + || Dy^k - D_y(X) - By^k ||_2^2 ) }
%
% SubProb. 2-1 : argmin_Dx = L(X^k+1, Dx, Dy^k, Bx^k, By^k)
% Dx^k+1 = argmin_Dx { lambda/2 * ( || Dx ||_2^2 ) + rho/2 * ( || Dx - D_x(X^k+1) - Bx^k ||_2^2 ) }
%
% SubProb. 2-2 : argmin_Dx = L(X^k+1, Dx^k+1, Dy, Bx^k, By^k)
% Dy^k+1 = argmin_Dy { lambda/2 * ( || Dy ||_2^2 ) + rho/2 * ( || Dy - D_y(X^k+1) - By^k ||_2^2 ) }
%
% SubProb. 3-1 : argmin_Bx = L(X^k+1, Dx^k+1, Dy^k+1, Bx, By^k)
% Bx^k+1 = Dx^k+1 - D_x(X^k+1) - Bx^k
%
% SubProb. 3-2 : argmin_Bx = L(X^k+1, Dx^k+1, Dy^k+1, Bx^k+1, By)
% By^k+1 = Dy^k+1 - D_y(X^k+1) - By^k

%%
clear ;
close all;
home;

%% GPU Processing
% If there is GPU device on your board, 
% then isgpu is true. Otherwise, it is false.
bgpu    = false;
bfig    = true;

%%  SYSTEM SETTING
N       = 512;
VIEW    = 360;
THETA   = linspace(0, 180, VIEW + 1);   THETA(end) = [];

R       = @(x) radon(x, THETA);
RT      = @(y) iradon(y, THETA, 'none', N)./(pi/(2*length(THETA)));
RINV    = @(y) iradon(y, THETA, N);

%% DATA GENERATION
load('XCAT512.mat');
x       = imresize(double(XCAT512), [N, N]);
p       = R(x);
x_full  = RINV(p);

%% LOW-DOSE SINOGRAM GENERATION
i0     	= 5e4;
pn     	= exp(-p);
pn     	= i0.*pn;
pn     	= poissrnd(pn);
pn      = max(-log(max(pn,1)./i0),0);

y       = pn;
x_low   = RINV(y);

%% -
LAMBDA  = 1e0;
RHO     = 5e3;

nADMM   = 1e1;
nCG     = 5e0;

CG_A    = @(x)      RT(R(x)) + RHO * ( Dxt(Dx(x)) + Dyt(Dy(x)) );

x_admm	= zeros(N);
Dx_     = zeros(N);
Dy_     = zeros(N);
Bx_     = zeros(N);
By_     = zeros(N);

RTY     = RT(y);

obj     = zeros(nADMM, 1);

L1              = @(x) norm(x, 1);
L2              = @(x) power(norm(x, 'fro'), 2);
COST.equation   = '1/2 * || A(X) - Y ||_2^2 + lambda * ( || D_x(X) ||_2^2 + || D_y(X) ||_2^2 )';
COST.function	= @(x) 1/2 * L2(R(x) - y) + LAMBDA/2 * (L2(Dx(x)) + L2(Dy(x)));

%% RUN ALTERNATING DIRECTION METHOD of MULTIFILERS (ADMM)
if bgpu
    x_admm 	= gpuArray(x_admm);
    Dx_     = gpuArray(Dx_);
    Dy_     = gpuArray(Dy_);
    Bx_     = gpuArray(Bx_);
    By_     = gpuArray(By_);
end

for iadmm = 1:nADMM

    % SOLVE SUBPROB. 1 for X using CG METHOD
    b       = RTY + RHO * ( Dxt(Dx_ - Bx_) + Dyt(Dy_ - By_) );
    x_admm	= CG(CG_A, b, x_admm, nCG, [], bfig);
    DxX_    = Dx(x_admm);
    DyX_    = Dy(x_admm);
    
    % SOLVE SUBPROB. 2 for Dx & Dy using SOFT THRESHOLDING 
    Dx_      = RHO/(LAMBDA + RHO) * (DxX_ + Bx_);
    Dy_      = RHO/(LAMBDA + RHO) * (DyX_ + By_);

    % SOLVE SUBPROB. 3 for Bx & By
    Bx_    	= Dx_ - DxX_ - Bx_;
    By_    	= Dy_ - DyX_ - By_;
    
    % CALCULATE COST
    obj(iadmm)  = COST.function(x_admm);
    
    % DISPLAY
    if bfig
        figure(1); colormap gray;
        subplot(121); imagesc(x_admm);      title([num2str(iadmm) ' / ' num2str(nADMM)]);
        subplot(122); semilogy(obj, '*-');  title(COST.equation);  xlabel('# of iteration');   ylabel('Objective'); 
                                            xlim([1, nADMM]);   grid on; grid minor;
        drawnow();
    end
end


%% CALCUATE QUANTIFICATION FACTOR
x_low       = max(x_low, 0);
x_admm      = max(x_admm, 0);
nor         = max(x(:));

mse_x_low   = immse(x_low./nor, x./nor);
mse_x_admm  = immse(x_admm./nor, x./nor);

psnr_x_low 	= psnr(x_low./nor, x./nor);
psnr_x_admm	= psnr(x_admm./nor, x./nor);

ssim_x_low  = ssim(x_low./nor, x./nor);
ssim_x_admm = ssim(x_admm./nor, x./nor);

%% DISPLAY
wndImg  = [0, 0.03];

figure('name', 'Alternating Direction Method of Multifliers (ADMM) Method for l2 regularization');
colormap(gray(256));

suptitle('Alternating Direction Method of Multifliers (ADMM) Method');
subplot(231);   imagesc(x,     	wndImg); 	axis image off;     title('ground truth');
subplot(232);   imagesc(x_full, wndImg);   	axis image off;     title(['full-dose_{FBP, view : ', num2str(VIEW) '}']);
subplot(234);   imagesc(x_low,  wndImg);   	axis image off;     title({['low-dose_{FBP, view : ', num2str(VIEW) '}'], ['MSE : ' num2str(mse_x_low, '%.4e')], ['PSNR : ' num2str(psnr_x_low, '%.4f')], ['SSIM : ' num2str(ssim_x_low, '%.4f')]});
subplot(235);   imagesc(x_admm, wndImg);  	axis image off;     title({['recon_{admm-l2}'], ['MSE : ' num2str(mse_x_admm, '%.4e')], ['PSNR : ' num2str(psnr_x_admm, '%.4f')], ['SSIM : ' num2str(ssim_x_admm, '%.4f')]});

subplot(2,3,[3,6]); semilogy(obj, '*-');    title(COST.equation);   xlabel('# of iteration');   ylabel('Objective'); 
                                            xlim([1, nADMM]);       grid on; grid minor;
