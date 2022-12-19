% demo method with all datasets for acc, NMI, purity, F-score

clc;
clear;
close all;
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));
addpath(genpath(pwd))

root_path = 'code_and_datasets/Saved_Results/compare_results/methods/';

% choose a dataset
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

nDim    = size(fea,2);
%--------------------------------------------------------------------------
Cluster = [2, 3, 4, 5, 6, 7, 8, 9, 10];
percent = 0.3;
alpha = 1;
nCase    = length(Cluster);  % 9
nRun     = 10; 


%--------------------------------------------------------------------------
acc_km     = zeros(nCase,nRun);
acc_nmf   = zeros(nCase,nRun);
acc_gnmf   = zeros(nCase,nRun);
acc_rsnmf   = zeros(nCase,nRun);
acc_snmfdsr   = zeros(nCase,nRun);
acc_cdcf    = zeros(nCase,nRun);
acc_cnmf   = zeros(nCase,nRun);
acc_lpcnmf = zeros(nCase,nRun);

nmi_km     = zeros(nCase,nRun);
nmi_nmf   = zeros(nCase,nRun);
nmi_gnmf   = zeros(nCase,nRun);
nmi_rsnmf   = zeros(nCase,nRun);
nmi_snmfdsr   = zeros(nCase,nRun);
nmi_cdcf    = zeros(nCase,nRun);
nmi_cnmf   = zeros(nCase,nRun);
nmi_lpcnmf = zeros(nCase,nRun);

pur_km     = zeros(nCase,nRun);
pur_nmf   = zeros(nCase,nRun);
pur_gnmf   = zeros(nCase,nRun);
pur_rsnmf   = zeros(nCase,nRun);
pur_snmfdsr   = zeros(nCase,nRun);
pur_cdcf    = zeros(nCase,nRun);
pur_cnmf   = zeros(nCase,nRun);
pur_lpcnmf = zeros(nCase,nRun);

fsc_km     = zeros(nCase,nRun);
fsc_nmf   = zeros(nCase,nRun);
fsc_gnmf   = zeros(nCase,nRun);
fsc_rsnmf   = zeros(nCase,nRun);
fsc_snmfdsr   = zeros(nCase,nRun);
fsc_cdcf    = zeros(nCase,nRun);
fsc_cnmf   = zeros(nCase,nRun);
fsc_lpcnmf = zeros(nCase,nRun);

ACC = zeros(9,8);
NMI = zeros(9,8);
PUR = zeros(9,8);
FSC = zeros(9,8);

fprintf('K \n');

tic;
caseIter = 0;
for k = Cluster   %2-10
    caseIter = caseIter + 1;
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
        
        for class = 1:k  %
            idx = find(gndSet == class);
            shuffleIndexes = randperm(length(idx));
            nSmpUnlabel = floor((percent)*length(idx));
            semiSplit(idx(shuffleIndexes(1:nSmpUnlabel))) = true;
        end
        
        nfeaSet = size(feaSet,1);  %   500k
        nSmpLabeled = sum(semiSplit); % 500k * percent(0.3)
        
        % shuffle the data sets and lables
        shuffleIndexes = randperm(nfeaSet); % 1 * 500k
        feaSet = feaSet(shuffleIndexes,:); % 500k * 784 shuffle
        gndSet = gndSet(shuffleIndexes); % 500k * 1 shuffle
        semiSplit = semiSplit(shuffleIndexes);  % 500k * 1 logical
        
        % constructing the similarity diagnal matrix based on labled data
        S = diag(semiSplit); % semiSplit(500k * 1 logical) diagonal metirx 500k * 500k
        
        % constructing the label constraint matrix for LpCNMF and CNMF
        E = eye(k); % k * k diagonal metirx
        A_mid = E(:,gndSet(semiSplit)); % k * (500k*percent(0.3))
        
        A_lpcnmf = zeros(k,nfeaSet); % k * 500k
        A_lpcnmf(:,semiSplit) = A_mid; % column with label turn to A_mid
        A_cnmf =[A_mid' zeros(nSmpLabeled, nfeaSet-nSmpLabeled); zeros(nfeaSet-nSmpLabeled,k) eye(nfeaSet - nSmpLabeled)];
        D1 = ones(k,nSmpLabeled); % k * (500k * percent(0.3))
        D_mid = D1(:,gndSet(semiSplit)); % k * (500k * percent(0.3))
        D2 = zeros(k,nfeaSet - nSmpLabeled); % k * (500k * 0.7)
        A_cdcf = [D_mid  D2]; % k * 500k 

        %Dissimilarity matrix
        d = zeros(nfeaSet, nfeaSet); % 500k * 500k
        for id=1:nSmpLabeled 
            for jd=id:nSmpLabeled 
                if id==jd
                    d(id,jd)=0;
                elseif  gndSet(id) == gndSet(jd)
                    d(id,jd)=0;
                else
                    d(id,jd)=1;
                end
            end
        end        
        D = d + d';

        %Similarity matrix
        s = zeros(nfeaSet, nfeaSet);
        for is=1:nSmpLabeled 
            for js=is:nSmpLabeled 
                if is==js
                    s(is,js)=1;
                elseif gndSet(is) == gndSet(js)
                    s(is,js)=1;
                else
                    s(is,js)=0;
                end
            end
        end
        S_bar = s + s';
        
        %------------------------------------------------------------------       
        %Clustering in the original space(K-means)
        label = litekmeans(feaSet,k,'Replicates',20); % 500k * 1
        pur = Purity(gndSet', label');
        label = bestMap(gndSet,label);
        acc = length(find(gndSet == label))/length(gndSet);
        fscore = Fscore(gndSet',label');
        nmi = MutualInfo(gndSet,label);

        acc_km(caseIter,runIter) = acc; 
        nmi_km(caseIter,runIter) = nmi;
        pur_km(caseIter,runIter) = pur; 
        fsc_km(caseIter,runIter) = fscore;

        %------------------------------------------------------------------
        %Clustering by Non-negative Matrix Factorization(NMF)
        options = [];
        options.WeightMode = 'Binary';
        options.NeighborMode = 'KNN';
        options.k = 5;
        options.maxIter = 200;
        W = constructW(feaSet,options);
        [~,Zest_nmf] = GNMF(feaSet', k, W, options);      
        label = litekmeans(Zest_nmf,k,'Replicates',20);
        pur = Purity(gndSet', label');
        label = bestMap(gndSet,label);
        acc = length(find(gndSet == label))/length(gndSet); 
        nmi = MutualInfo(gndSet,label);
        fscore = Fscore(gndSet',label');

        acc_nmf(caseIter,runIter) = acc; 
        nmi_nmf(caseIter,runIter) = nmi;
        pur_nmf(caseIter,runIter) = pur; 
        fsc_nmf(caseIter,runIter) = fscore;

        %------------------------------------------------------------------
        %Clustering by Graph regularized Non-negative Matrix Factorization (GNMF)
        options = [];
        options.WeightMode = 'HeatKernel';
        options.NeighborMode = 'KNN';
        options.k = 5;
        options.t = 1;
        options.maxIter = 200;
        W = constructW(feaSet,options);
        [~,Zest_gnmf] = GNMF(feaSet', k, W, options);      
        label = litekmeans(Zest_gnmf,k,'Replicates',20);
        pur = Purity(gndSet', label');
        label = bestMap(gndSet,label);
        acc = length(find(gndSet == label))/length(gndSet);
        nmi = MutualInfo(gndSet,label);
        fscore = Fscore(gndSet',label');

        acc_gnmf(caseIter,runIter) = acc; 
        nmi_gnmf(caseIter,runIter) = nmi;
        pur_gnmf(caseIter,runIter) = pur;
        fsc_gnmf(caseIter,runIter) = fscore;

        %------------------------------------------------------------------        
        %Clustering by RSNMF 
        options = [];     
        options.WeightMode = 'Binary';
        options.NeighborMode = 'KNN';
        options.k = 5;  
        options.maxIter = 200;
        p = 2;
        m = 2;
        [~, Zest_Rsnmf] = RSNMF(feaSet', k, p, m, nfeaSet, nSmpLabeled, options);
        label = litekmeans(Zest_Rsnmf', k, 'Replicates',20); 
        pur = Purity(gndSet', label');
        label = bestMap(gndSet,label);
        acc = length(find(gndSet(~semiSplit) == label(~semiSplit)))/length(gndSet(~semiSplit));
        nmi = MutualInfo(gndSet(~semiSplit),label(~semiSplit));
        fscore = Fscore(gndSet',label');

        acc_rsnmf(caseIter,runIter) = acc;    
        nmi_rsnmf(caseIter,runIter) = nmi;
        pur_rsnmf(caseIter,runIter) = pur;
        fsc_rsnmf(caseIter,runIter) = fscore;

        %------------------------------------------------------------------       
        %Clustering by SNMFDSR 
        options = [];     
        options.WeightMode = 'HeatKernel';
        options.t = 1; 
        options.NeighborMode = 'KNN';
        options.k = 5;  
        options.maxIter = 200;
        W = constructW(feaSet,options);
        SS = S_bar + W;
        [~, Zest_SNMFDSR] = SNMFDSR(feaSet', D, SS, k, options);
        label = litekmeans(Zest_SNMFDSR', k, 'Replicates',20); 
        pur = Purity(gndSet', label');
        label = bestMap(gndSet,label);
        acc = length(find(gndSet(~semiSplit) == label(~semiSplit)))/length(gndSet(~semiSplit));
        nmi = MutualInfo(gndSet(~semiSplit),label(~semiSplit));
        fscore = Fscore(gndSet',label');

        acc_snmfdsr(caseIter,runIter) = acc;
        nmi_snmfdsr(caseIter,runIter) = nmi;
        pur_snmfdsr(caseIter,runIter) = pur;
        fsc_snmfdsr(caseIter,runIter) = fscore;
        
        %------------------------------------------------------------------
        %Clustering by Class Driven Concept Factorization (CDCF) 
        options = [];
        options.WeightMode = 'Binary';
        options.NeighborMode = 'KNN';
        options.k = 5;
        options.maxIter = 200;
        W = constructW(feaSet,options);
        [~,Zest_cdcf] = CDCF(feaSet',A_cdcf, k, W, options);
        label = litekmeans(Zest_cdcf, k,'Replicates',20); 
        pur = Purity(gndSet', label');
        label = bestMap(gndSet,label);
        acc = length(find(gndSet(~semiSplit) == label(~semiSplit)))/length(gndSet(~semiSplit)); 
        nmi = MutualInfo(gndSet(~semiSplit),label(~semiSplit));
        fscore = Fscore(gndSet',label');

        acc_cdcf(caseIter,runIter) = acc; 
        nmi_cdcf(caseIter,runIter) = nmi;
        pur_cdcf(caseIter,runIter) = pur;
        fsc_cdcf(caseIter,runIter) = fscore;

        %------------------------------------------------------------------
        %Clustering by Constrained Non-negative Matrix Factorization(CNMF)
        options = [];
        options.WeightMode = 'HeatKernel';
        options.NeighborMode = 'KNN';
        options.k = 5;
        options.t = 1;
        options.maxIter = 200;
        W = constructW(feaSet,options);
        [~,Zest_cnmf] = CNMF(feaSet',A_cnmf, k, W, options); 
        Zest_cnmf = A_cnmf*Zest_cnmf;
        label = litekmeans(Zest_cnmf, k,'Replicates',20); 
        pur = Purity(gndSet', label');
        label = bestMap(gndSet,label);
        acc = length(find(gndSet(~semiSplit) == label(~semiSplit)))/length(gndSet(~semiSplit)); 
        nmi = MutualInfo(gndSet(~semiSplit),label(~semiSplit));
        fscore = Fscore(gndSet',label');

        acc_cnmf(caseIter,runIter) = acc;
        nmi_cnmf(caseIter,runIter) = nmi;
        pur_cnmf(caseIter,runIter) = pur;
        fsc_cnmf(caseIter,runIter) = fscore;


        %--------------------------------------------------------------------------------------
        %Clustering by Label Propagation Constrained Non-negative Matrix Factorization(LpCNMF)
        options = [];
        options.WeightMode = 'HeatKernel';
        options.NeighborMode = 'KNN';
        options.k = 5;
        options.t = 1;
        options.maxIter = 200;
        W = constructW(feaSet,options);
        [~, Zest_lpcnmf, F] = LpCNMF(feaSet', A_lpcnmf', k, k, S, W, options); 
        Zest_lpcnmf = F*Zest_lpcnmf;        
        label = litekmeans(Zest_lpcnmf, k, 'Replicates',20);
        pur = Purity(gndSet', label');
        label = bestMap(gndSet,label);
        acc = length(find(gndSet(~semiSplit) == label(~semiSplit)))/length(gndSet(~semiSplit)); 
        nmi = MutualInfo(gndSet(~semiSplit),label(~semiSplit));
        fscore = Fscore(gndSet',label');

        acc_lpcnmf(caseIter,runIter) = acc; 
        nmi_lpcnmf(caseIter,runIter) = nmi;
        pur_lpcnmf(caseIter,runIter) = pur; 
        fsc_lpcnmf(caseIter,runIter) = fscore;
    end
    
    fprintf('%d \n',k);

    acc=[mean(acc_km(caseIter,:)),mean(acc_nmf(caseIter,:)),mean(acc_gnmf(caseIter,:)),...
        mean(acc_rsnmf(caseIter,:)),mean(acc_snmfdsr(caseIter,:)),mean(acc_cdcf(caseIter,:)),...
        mean(acc_cnmf(caseIter,:)),mean(acc_lpcnmf(caseIter,:))];    
    nmi=[mean(nmi_km(caseIter,:)),mean(nmi_nmf(caseIter,:)),mean(nmi_gnmf(caseIter,:)),...
        mean(nmi_rsnmf(caseIter,:)),mean(nmi_snmfdsr(caseIter,:)),mean(nmi_cdcf(caseIter,:)),...
        mean(nmi_cnmf(caseIter,:)),mean(nmi_lpcnmf(caseIter,:))];
    pur=[mean(pur_km(caseIter,:)),mean(pur_nmf(caseIter,:)),mean(pur_gnmf(caseIter,:)),...
        mean(pur_rsnmf(caseIter,:)),mean(pur_snmfdsr(caseIter,:)),mean(pur_cdcf(caseIter,:)),...
        mean(pur_cnmf(caseIter,:)),mean(pur_lpcnmf(caseIter,:))]; 
    fsc=[mean(fsc_km(caseIter,:)),mean(fsc_nmf(caseIter,:)),mean(fsc_gnmf(caseIter,:)),...
        mean(fsc_rsnmf(caseIter,:)),mean(fsc_snmfdsr(caseIter,:)),mean(fsc_cdcf(caseIter,:)),...
        mean(fsc_cnmf(caseIter,:)),mean(fsc_lpcnmf(caseIter,:))];
    ACC(caseIter,:) = acc;
    NMI(caseIter,:) = nmi;
    PUR(caseIter,:) = pur;
    FSC(caseIter,:) = fsc;

    %% save method results
    aim_root_path = [root_path, dataset,'/'];
    if ~exist(aim_root_path,'dir')==1
        mkdir(aim_root_path);
    end
    
    acc_file_names = {'acc_cdcf','acc_cnmf','acc_gnmf','acc_km',...
        'acc_lpcnmf','acc_nmf','acc_rsnmf','acc_snmfdsr'};
    nmi_file_names = {'nmi_cdcf','nmi_cnmf','nmi_gnmf','nmi_km',...
        'nmi_lpcnmf','nmi_nmf','nmi_rsnmf','nmi_snmfdsr'};
    pur_file_names = {'pur_cdcf','pur_cnmf','pur_gnmf','pur_km',...
        'pur_lpcnmf','pur_nmf','pur_rsnmf','pur_snmfdsr'};
    fsc_file_names = {'fsc_cdcf','fsc_cnmf','fsc_gnmf','fsc_km',...
        'fsc_lpcnmf','fsc_nmf','fsc_rsnmf','fsc_snmfdsr'};
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
