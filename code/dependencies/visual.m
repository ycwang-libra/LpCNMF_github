function visual(W, mag, cols, ysize)
%--------------------------------------------------------------------------
% visual - display a basis for image patches
% V = W * H;
%           W: the basis, with patches as column vectors
%           mag: magnification factor
%          cols: number of columns (x-dimension of map)
%          ysize: [optional] height of each subimage
%--------------------------------------------------------------------------
% Is the basis non-negative
sep = 2;
maxi = max(W);
mini = min(W);
bgval = 1;
W = (W - repmat(mini,[size(W,1) 1]))./...
(repmat(maxi,[size(W,1) 1])-repmat(mini,[size(W,1) 1]));

% Get maximum absolute value (it represents white or black; zero is gray)
%--------------------------------------------------------------------------
% the size of basis images
if ~exist('ysize','var'), ysize = sqrt(size(W,1)); end
xsize = size(W,1)/ysize;
%--------------------------------------------------------------------------
 xsizep = xsize+sep; ysizep = ysize+sep;
%--------------------------------------------------------------------------
% define the size of maps
rows = ceil(size(W,2)/cols);
%--------------------------------------------------------------------------
% whole size of the maps of image
I = bgval*ones(2+ysize*rows+(rows-1)*sep,2+xsize*cols+(cols-1)*sep);
%--------------------------------------------------------------------------
for i=0:rows-1
    for j=0:cols-1
        if i*cols+j+1>size(W,2)
            % This leaves it at background color   
        else
            % This sets the patch
            I(i*ysizep+2:i*ysizep+ysize+1, ...
                j*xsizep+2:j*xsizep+xsize+1) = ...
                reshape(W(:,i*cols+j+1),[ysize xsize]);
        end
    end
end
%--------------------------------------------------------------------------
I = imresize(I,mag);
%--------------------------------------------------------------------------
%colormap(gray(256));
iptsetpref('ImshowBorder','tight');
subplot('position',[0,0,1,1]);
imshow(I,[ ]);
%imagesc(I); axis off; axis equal; axis tight
%imshow(-I,[-maxi -mini]);
