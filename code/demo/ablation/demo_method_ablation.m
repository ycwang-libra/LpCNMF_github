% demo ablation for label propagation with all datasets for acc, NMI, purity, F-score

clc;
clear;
close all;
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));
addpath(genpath(pwd))

root_path = 'code_and_datasets/Saved_Results/ablation_results_label_propagation/';

% choose a dataset
% {'AR','COIL20','COIL100','MNIST','umist','USPS','Yale_32x32','YaleB_32x32'}
dataset = 'MNIST';

switch dataset
    case 'MNIST'  % 70000 sample  784 feature_dim 
        load([dataset,'.mat']);  % fea 70000 x 784  gnd  70000 x 1  10classes 1class about 7000samples
        nEach  = 500;               % number of sample in every class is 500

    case 'AR'     % 700   sample  19800 feature_dim
        load([dataset,'.mat']);  % Tr_dataMatrix 19800 x 700  gnd  1 x 700 100classes 1class about 7samples
        gnd = Tr_sampleLabels';  % 700 x 1  
        fea = Tr_dataMatrix';     % 700 x  19800
        nEach = 7; 

    case 'umist'  % 575   sample  644 feature_dim
        load([dataset,'.mat']);  % X    644 x 575   gnd    575 x 1   20classes 1class about 25 samples
        fea = X';                % 575 x 644
        nEach    = 19;

    case 'USPS'   % 9298  sample  256 feature_dim
        load([dataset,'.mat']);  % fea 9298 x 256  gnd  9298 x 1   10classes 1class about 1000 samples
        fea = Normalize255(fea); % -1~ 1 --> 0~255
        nEach    = 708;
        
    case 'COIL20' % 1440   sample   1024  feature_dim
        load([dataset,'.mat']);  % fea 1440 x 1024  gnd  1440 x 1   20classes 1class 72 samples
        nEach    = 72;

    case 'COIL100' % 7200   sample   1024  feature_dim
        load([dataset,'.mat']);  % fea 7200 x 1024  gnd  7200 x 1   100classes 1class 72 samples
        nEach    = 72;

    case 'Yale_32x32' % 165   sample   1024  feature_dim
        load([dataset,'.mat']);  % fea 165 x 1024  gnd  165 x 1   15classes 1class 15 samples
        nEach    = 11; 

    case 'YaleB_32x32' % 2414   sample   1024  feature_dim
        load([dataset,'.mat']);  % fea 2414 x 1024  gnd  2414 x 1   38classes 1class about 64 samples
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
Cluster = [2, 3, 4, 5, 6, 7, 8, 9, 10];
percent = 0.3;
nCase    = length(Cluster);
nRun     = 10;

%--------------------------------------------------------------------------
acc_lpcnmf_wo_lp = zeros(nCase,nRun);
nmi_lpcnmf_wo_lp = zeros(nCase,nRun);
pur_lpcnmf_wo_lp = zeros(nCase,nRun);
fsc_lpcnmf_wo_lp = zeros(nCase,nRun);

ACC = zeros(9,1); % num_cluters
NMI = zeros(9,1);
PUR = zeros(9,1);
FSC = zeros(9,1);

fprintf('K  acc nmi  purity f-score \n');

tic;
caseIter = 0;
for k = Cluster   %2-10
    caseIter = caseIter + 1; % number of cluster
    nSample  = nEach*k;
    PercentIter = 0;
    for runIter = 1:nRun 
        index  = 0;
        Samples = zeros(nSample,nDim);
        labels  = zeros(nSample,1);
        
        shuffleClasses = randperm(nClass);
        if strcmp(dataset,'MNIST')
            I = ones(1,10);
            shuffleClasses = shuffleClasses - I;   % label for MNIST is 0-9
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
        
        for class = 1:k
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
        S = diag(semiSplit); % semiSplit(500k * 1 logical) diagonal metirx 500k * 500k
        
        % constructing the label constraint matrix for LpCNMF and CNMF
        E = eye(k); % k * k diagonal metirx
        A_mid = E(:,gndSet(semiSplit)); % k * (500k*percent(0.3))
        
        A_lpcnmf = zeros(k,nfeaSet); % k * 500k
        A_lpcnmf(:,semiSplit) = A_mid; % column with label turn to A_mid

        %--------------------------------------------------------------------------------------
        %Clustering by LpCNMF without Lp
        options = [];
        options.WeightMode = 'HeatKernel';
        options.NeighborMode = 'KNN';
        options.k = 5;
        options.t = 1;
        options.maxIter = 200;
        W = constructW(feaSet,options);
        [~, Zest_lpcnmf, F] = LpCNMF_wo_Lp(feaSet', A_lpcnmf', k, k, S, W, options); 
        Zest_lpcnmf = F*Zest_lpcnmf;        
        label = litekmeans(Zest_lpcnmf, k, 'Replicates',20);
        pur = Purity(gndSet', label');
        label = bestMap(gndSet,label);
        acc = length(find(gndSet(~semiSplit) == label(~semiSplit)))/length(gndSet(~semiSplit)); 
        nmi = MutualInfo(gndSet(~semiSplit),label(~semiSplit));
        fscore = Fscore(gndSet',label');
        acc_lpcnmf_wo_lp(caseIter,runIter) = acc; 
        nmi_lpcnmf_wo_lp(caseIter,runIter) = nmi;
        pur_lpcnmf_wo_lp(caseIter,runIter) = pur; 
        fsc_lpcnmf_wo_lp(caseIter,runIter) = fscore;
    end
    
    fprintf('%d   %3.3f   %3.3f   %3.3f    %3.3f \n', k, mean(acc_lpcnmf_wo_lp(caseIter,:)), ...
            mean(nmi_lpcnmf_wo_lp(caseIter,:)), mean(pur_lpcnmf_wo_lp(caseIter,:)), mean(fsc_lpcnmf_wo_lp(caseIter,:)));

    ACC(caseIter,:)=mean(acc_lpcnmf_wo_lp(caseIter,:));
    NMI(caseIter,:)=mean(nmi_lpcnmf_wo_lp(caseIter,:));
    PUR(caseIter,:)=mean(pur_lpcnmf_wo_lp(caseIter,:));
    FSC(caseIter,:)=mean(fsc_lpcnmf_wo_lp(caseIter,:));

    %% save method results
    aim_root_path = [root_path, dataset,'/'];
    if ~exist(aim_root_path,'dir')==1
        mkdir(aim_root_path);
    end

    acc_file_names = {'acc_lpcnmf_wo_lp'};
    nmi_file_names = {'nmi_lpcnmf_wo_lp'};
    pur_file_names = {'pur_lpcnmf_wo_lp'};
    fsc_file_names = {'fsc_lpcnmf_wo_lp'};
    num_methods = length(acc_file_names);
    for i = 1:num_methods
        acc_path = [aim_root_path, acc_file_names{i}];
        save(acc_path, acc_file_names{i});
        nmi_path = [aim_root_path, nmi_file_names{i}];
        save(nmi_path, nmi_file_names{i});
        pur_path = [aim_root_path, pur_file_names{i}];
        save(pur_path, pur_file_names{i});
        fsc_path = [aim_root_path, fsc_file_names{i}];
        save(fsc_path, fsc_file_names{i});
    end

end
toc;
