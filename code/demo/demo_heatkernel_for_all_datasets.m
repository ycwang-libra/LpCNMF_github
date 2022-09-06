% demo heatkernel t with all datasets for acc, NMI, purity, F-score
clc;
clear;
close all;
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));
addpath(genpath(pwd))

root_path = 'code_and_datasets/Saved_Results/heatkernel_results/';

% choose a dataset
% {'AR','COIL20','COIL100','MNIST','UMIST','USPS','Yale','YaleB'}
dataset = 'UMIST';

switch dataset
    case 'MNIST'  % 70000 sample  784 feature_dim 
        load('MNIST.mat');  % fea 70000 x 784  gnd  70000 x 1  10classes 1class about 7000samples
        nEach  = 500;               % number of sample in every class is 500

    case 'AR'     % 700   sample  19800 feature_dim
        load('AR.mat');  % Tr_dataMatrix 19800 x 700  gnd  1 x 700 100classes 1 class about 7samples
        gnd = Tr_sampleLabels';  % 700 x 1  
        fea = Tr_dataMatrix';     % 700 x  19800
        nEach = 7; 

    case 'UMIST'  % 575   sample  644 feature_dim
        load('umist.mat');  % X    644 x 575   gnd    575 x 1   20classes 1 class about 25 samples
        fea = X';                % 575 x 644
        nEach    = 19;

    case 'USPS'   % 9298  sample  256 feature_dim
        load('USPS.mat');  % fea 9298 x 256  gnd  9298 x 1   10classes 1 class about 1000 samples
        fea = Normalize255(fea); % -1~ 1 --> 0~255
        nEach    = 708;
        
    case 'COIL20' % 1440   sample   1024  feature_dim
        load('COIL20.mat');  % fea 1440 x 1024  gnd  1440 x 1   20classes 1 class 72 samples
        nEach    = 72;

    case 'COIL100' % 7200   sample   1024  feature_dim
        load('COIL100.mat');  % fea 7200 x 1024  gnd  7200 x 1   100classes 1 class 72 samples
        nEach    = 72;

    case 'Yale' % 165   sample   1024  feature_dim
        load('Yale_32x32.mat');  % fea 165 x 1024  gnd  165 x 1   15classes 1 class 15 samples
        nEach    = 11; 

    case 'YaleB' % 2414   sample   1024  feature_dim
        load('YaleB_32x32.mat');  % fea 2414 x 1024  gnd  2414 x 1   38classes 1 class about 64 samples
        nEach    = 59;
end

nClass  = length(unique(gnd));   %nClass

%normalization the feature
if strcmp(dataset,'COIL100') || strcmp(dataset,'AR')
    fea = NormalizeFea(double(fea));
else
    fea = NormalizeFea(fea);
end

nDim    = size(fea,2);
%--------------------------------------------------------------------------
nCluster = [2 3 4 5 6 7 8 9 10];
Range_t=[0.001,0.01,0.1,1,10,100,1000];
percent = 0.3;
alpha = 1;
nCase  = length(nCluster);
num_heatkernel = length(Range_t);
nRun = 10; 
maxIter = 100;

ACC=zeros(nCase,num_heatkernel,nRun);
NMI=zeros(nCase,num_heatkernel,nRun);
PUR=zeros(nCase,num_heatkernel,nRun);
FSC=zeros(nCase,num_heatkernel,nRun);

tic;
fprintf('HeatKernel  0.001   0.01     0.1       1     10      100    1000 \n');

caseIter = 0;
for k = nCluster
    caseIter = caseIter + 1;
    nSample  = nEach*k;
    fprintf('%d ',k);
    rIter = 0;
    for t = Range_t
        rIter = rIter + 1;
        for runIter = 1:nRun     %nRun=1~10        
            index  = 0;
            Samples = zeros(nSample,nDim);
            labels  = zeros(nSample,1);
            shuffleClasses = randperm(nClass);
            if strcmp(dataset,'MNIST')
                I = ones(1,10);
                shuffleClasses = shuffleClasses - I;   % label for MNIST is 0-9
            end
            
            for class = 1:k      % 1~（2~10）
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
            
            nfeaSet = size(feaSet,1);
            nSmpLabeled = sum(semiSplit);
            
            % shuffle the data sets and lables
            shuffleIndexes = randperm(nfeaSet); 
            feaSet = feaSet(shuffleIndexes,:);
            gndSet = gndSet(shuffleIndexes);
            semiSplit = semiSplit(shuffleIndexes);
            
            % constructing the similarity diagnal matrix based on labled data
            S = diag(semiSplit);
            
            % constructing the label constraint matrix for LpCNMF and CNMF
            E = eye(k);
            A_mid = E(:,gndSet(semiSplit)); % k * (500k*percent(0.3))
            
            A_lpcnmf = zeros(k,nfeaSet); % k * 500k
            A_lpcnmf(:,semiSplit) = A_mid; % column with label turn to A_mid
            
            %--------------------------------------------------------------------------------------
            %Clustering by Label Propagation Constrained Non-negative Matrix Factorization(LpCNMF)
            options = [];
            options.WeightMode = 'HeatKernel';
            options.NeighborMode = 'KNN';
            options.k = 5;   
            options.t = t;
            options.maxIter = maxIter;
            W = constructW(feaSet,options);
            [~, Zest_lpcnmf, F] = LpCNMF(feaSet', A_lpcnmf', k, k, S, W, options);
            Zest_lpcnmf = F*Zest_lpcnmf;
            label = litekmeans(Zest_lpcnmf, k, 'Replicates',20); 
            pur = Purity(gndSet', label');
            label = bestMap(gndSet,label);
            acc = length(find(gndSet(~semiSplit) == label(~semiSplit)))/length(gndSet(~semiSplit));
            nmi = MutualInfo(gndSet(~semiSplit),label(~semiSplit));
            fscore = Fscore(gndSet',label');

            ACC(caseIter,rIter,runIter) = acc;
            NMI(caseIter,rIter,runIter) = nmi;
            PUR(caseIter,rIter,runIter) = pur; 
            FSC(caseIter,rIter,runIter) = fscore;
        end
        fprintf('   %3.3f',mean(ACC(caseIter,rIter,:)));   
    end
    fprintf('\n'); 
end
toc;

ACC_mean = squeeze(mean(ACC,3));
% calculate mean value and find optimal
Ave_ac = mean(ACC_mean,1);
fprintf('AVE  ');
fprintf(' %3.3f  ',Ave_ac);
fprintf('\n');
fprintf('best_r  ');
fprintf('%3.3f ', Range_t(Ave_ac==max(Ave_ac)));

figure(1)
x=[1 2 3 4 5 6 7];
plot(x,ACC_mean(1,:),'-ok',x,ACC_mean(2,:),'-ob',  x,ACC_mean(3,:),'-og',  x,ACC_mean(4,:),'-om', ...
    x,ACC_mean(5,:),'-oc',  x,ACC_mean(6,:),'-or', x,ACC_mean(7,:),'-vg',  x,ACC_mean(8,:),'-vr',  ...
    x,ACC_mean(9,:),'-vm');
set(gca,'XTickLabel',{'0.001' '0.01' '0.1' '1' '10' '100' '1000'})
legend('k=2','k=3','k=4','k=5','k=6','k=7','k=8','k=9','k=10');
xlabel('HeatKernel Factor t');
ylabel('ACC');
title([dataset,'-HeatKernel']);
 
% save the obj
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
