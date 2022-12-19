
clc;
clear;
close all;
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));
addpath(genpath(pwd))

root_path = 'code_and_datasets/Saved_Results/Add_results/labelproportion_with_heat_kernel/';

% {'AR','COIL20','COIL100','MNIST','UMIST','USPS','Yale','YaleB'}
dataset = 'UMIST';

switch dataset
    case 'MNIST'  % 70000 sample  784 feature_dim 
        load('MNIST.mat');  % fea 70000 x 784  gnd  70000 x 1  10classes 1class about 7000samples
        nEach  = 500;               % number of sample in every class is 500

    case 'AR'     % 700   sample  19800 feature_dim
        load('AR.mat');  % Tr_dataMatrix 19800 x 700  gnd  1 x 700 100classes 1class about 7samples
        gnd = Tr_sampleLabels';  % 700 x 1  
        fea = Tr_dataMatrix';     % 700 x  19800
        nEach = 7; 

    case 'UMIST'  % 575   sample  644 feature_dim
        load('umist.mat');  % X    644 x 575   gnd    575 x 1   20classes 1class about 25 samples
        fea = X';                % 575 x 644
        nEach    = 19;

    case 'USPS'   % 9298  sample  256 feature_dim
        load('USPS.mat');  % fea 9298 x 256  gnd  9298 x 1   10classes 1class about 1000 samples
        fea = Normalize255(fea); % -1~ 1 --> 0~255
        nEach    = 708;
        
    case 'COIL20' % 1440   sample   1024  feature_dim
        load('COIL20.mat');  % fea 1440 x 1024  gnd  1440 x 1   20classes 1class 72 samples
        nEach    = 72;

    case 'COIL100' % 7200   sample   1024  feature_dim
        load('COIL100.mat');  % fea 7200 x 1024  gnd  7200 x 1   100classes 1class 72 samples
        nEach    = 72;

    case 'Yale' % 165   sample   1024  feature_dim
        load('Yale_32x32.mat');  % fea 165 x 1024  gnd  165 x 1   15classes 1class 15 samples
        nEach    = 11; 

    case 'YaleB' % 2414   sample   1024  feature_dim
        load('YaleB_32x32.mat');  % fea 2414 x 1024  gnd  2414 x 1   38classes 1class about 64 samples
        nEach    = 59;
end

nClass  = length(unique(gnd));   %nClass

%normalization the feature
if strcmp(dataset,'COIL100') || strcmp(dataset,'AR')
    fea = NormalizeFea(double(fea));
else
    fea = NormalizeFea(fea);
end

nDim    = size(fea,2);           %nDim = 28 x 28 = 784
%--------------------------------------------------------------------------
Cluster = [2, 3, 4, 5, 6, 7, 8, 9, 10];

Label_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
Heat_kernel = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
nCase    = length(Cluster);  % 9
nPercent = length(Label_percent); % 10
nKernel = length(Heat_kernel); % 7
nRun     = 5;                % every experiment are conducted for 5 times


%--------------------------------------------------------------------------
ACC = zeros(nCase, nPercent, nKernel, nRun); % 9 x 9 x 7 x 5
NMI = zeros(nCase, nPercent, nKernel, nRun); % 9 x 9 x 7 x 5
PUR = zeros(nCase, nPercent, nKernel, nRun); % 9 x 9 x 7 x 5
FSC = zeros(nCase, nPercent, nKernel, nRun); % 9 x 9 x 7 x 5

fprintf('Label_percent  0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9\n');

tic;
ClusterIter = 0;
for k = Cluster   %2-10
    ClusterIter = ClusterIter + 1;
    nSample  = nEach*k;
    PercentIter = 0;
    for percent = Label_percent
        PercentIter = PercentIter + 1;
        KernelIter = 0;
        for t = Heat_kernel
            KernelIter = KernelIter + 1;
            for runIter = 1:nRun     %nRun=1~10        
                index  = 0;
                Samples = zeros(nSample,nDim);
                labels  = zeros(nSample,1);
                
                shuffleClasses = randperm(nClass); 
                if strcmp(dataset,'MNIST')
                    I = ones(1,10);
                    shuffleClasses = shuffleClasses - I;
                end
                
                for class = 1:k
                    idx = find(gnd == shuffleClasses(class)); 
                    sampleEach = fea(idx(1:nEach),:);
                    Samples(index+1:index+nEach,:) = sampleEach; 
                    labels(index+1:index+nEach,:)  = class; 
                    index = index + nEach;
                end
                
                %------------------------------------------------------------------
                feaSet = Samples; 
                gndSet = labels; 
                semiSplit = false(size(gndSet));  
                
                for class = 1:k  %
                    idx = find(gndSet == class); 
                    shuffleIndexes = randperm(length(idx)); 
                    nSmpUnlabel = floor((percent)*length(idx));
                    semiSplit(idx(shuffleIndexes(1:nSmpUnlabel))) = true; 
                end
                
                nfeaSet = size(feaSet,1);  %   500k
                nSmpLabeled = sum(semiSplit); % 500k * percent(0.3)
                
                % shuffle the data sets and lables
                shuffleIndexes = randperm(nfeaSet);
                feaSet = feaSet(shuffleIndexes,:); 
                gndSet = gndSet(shuffleIndexes); 
                semiSplit = semiSplit(shuffleIndexes);
                
                % constructing the similarity diagnal matrix based on labled data
                S = diag(semiSplit);
                
                % constructing the label constraint matrix for LpCNMF and CNMF
                E = eye(k); % k * k
                A_mid = E(:,gndSet(semiSplit)); % k * (500k*percent(0.3)) 
                
                A_lpcnmf = zeros(k,nfeaSet);
                A_lpcnmf(:,semiSplit) = A_mid; 
                
                %--------------------------------------------------------------------------------------
                %Clustering by Label Propagation Constrained Non-negative Matrix Factorization(LpCNMF)
                options = [];
                options.WeightMode = 'HeatKernel';
                options.NeighborMode = 'KNN';
                options.k = 5;
                options.t = t;
                options.maxIter = 200; 
                W = constructW(feaSet,options);
                [~, Zest_lpcnmf, F] = LpCNMF(feaSet', A_lpcnmf', k, k, S, W, options); 
                Zest_lpcnmf = F*Zest_lpcnmf;        
                label = litekmeans(Zest_lpcnmf, k, 'Replicates',20); 
                pur = Purity(gndSet', label');
                label = bestMap(gndSet,label);
                acc = length(find(gndSet(~semiSplit) == label(~semiSplit)))/length(gndSet(~semiSplit)); 
                nmi = MutualInfo(gndSet(~semiSplit),label(~semiSplit));
                fsc = Fscore(gndSet',label');

                ACC(ClusterIter,PercentIter,KernelIter,runIter) = acc; 
                NMI(ClusterIter,PercentIter,KernelIter,runIter) = nmi;
                PUR(ClusterIter,PercentIter,KernelIter,runIter) = pur; 
                FSC(ClusterIter,PercentIter,KernelIter,runIter) = fsc;
            end
        end
    end
    
    fprintf('%d   %3.3f   %3.3f   %3.3f    %3.3f   %3.3f   %3.3f   %3.3f   %3.3f   %3.3f \n',...
        k, mean(ACC(ClusterIter,1,4,:)), mean(ACC(ClusterIter,2,4,:)), ...
        mean(ACC(ClusterIter,3,4,:)), mean(ACC(ClusterIter,4,4,:)), ...
        mean(ACC(ClusterIter,5,4,:)), mean(ACC(ClusterIter,6,4,:)), ...
        mean(ACC(ClusterIter,7,4,:)), mean(ACC(ClusterIter,8,4,:)), ...
        mean(ACC(ClusterIter,9,4,:)));
    
    aim_root_path = [root_path, dataset,'/'];
    if ~exist(aim_root_path,'dir')==1
        mkdir(aim_root_path);
    end
    acc_aim_path = [aim_root_path, 'ACC'];
    nmi_aim_path = [aim_root_path, 'NMI'];
    pur_aim_path = [aim_root_path, 'PUR'];
    fsc_aim_path = [aim_root_path, 'FSC'];
    save(acc_aim_path, 'ACC');
    save(nmi_aim_path, 'NMI');
    save(pur_aim_path, 'PUR');
    save(fsc_aim_path, 'FSC');

end
toc;


