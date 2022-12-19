
clc;
clear;
close all;
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',1));
addpath(genpath(pwd))
aim_root_path = 'code_and_datasets\Saved_Results\labelproportion_with_regularization';

% {'AR','COIL20','COIL100','MNIST','umist','USPS','Yale_32x32','YaleB_32x32'}
dataset = 'MNIST';

switch dataset
    case 'MNIST'  % 70000 sample  784 feature_dim 
        load([dataset,'.mat']);  % fea 70000 x 784  gnd  70000 x 1  10classes 1class about 7000samples

        nEach  = 500;               % number of sample in every class is 500

    case 'AR'     % 700   sample  19800 feature_dim
        load([dataset,'.mat']);  % Tr_dataMatrix 19800 x 700  gnd  1 x 700 100classes 1class about 7samples
        gnd = Tr_sampleLabels';  % 700 x 1  
%         fea = Tr_dataMatrix;     % 19800 x 700 ???   Tr_dataMatrix' 700 x 19800
        fea = Tr_dataMatrix';     %  700 x  19800

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

nDim    = size(fea,2);           %nDim = 28 x 28 = 784
%--------------------------------------------------------------------------
Cluster = [2, 3, 4, 5, 6, 7, 8, 9, 10];
Label_percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
Regular = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
nCluster    = length(Cluster);  % 9
nPercent = length(Label_percent); % 10
nRegular = length(Regular); % 7
nRun     = 20;


%--------------------------------------------------------------------------
acc_lpcnmf = zeros(nCluster, nPercent, nRegular, nRun); % 9 x 9 x 7 x 5
nmi_lpcnmf = zeros(nCluster, nPercent, nRegular, nRun); % 9 x 9 x 7 x 5

fprintf('Label_percent  0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9\n');

tic;
ClusterIter = 0;
for k = Cluster   %2-10
    ClusterIter = ClusterIter + 1;
    nSample  = nEach*k;
    PercentIter = 0;
    for percent = Label_percent
        PercentIter = PercentIter + 1;
        RegularIter = 0;
        for r = Regular
            RegularIter = RegularIter + 1;
            for runIter = 1:nRun    
                index  = 0;
                Samples = zeros(nSample,nDim);
                labels  = zeros(nSample,1);
                
                shuffleClasses = randperm(nClass);
                if strcmp(dataset,'MNIST')
                    I = ones(1,10);
                    shuffleClasses = shuffleClasses - I;
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
                E = eye(k); % k * k
                A_mid = E(:,gndSet(semiSplit)); % k * (500k*percent(0.3))
                
                A_lpcnmf = zeros(k,nfeaSet); % k * 500k
                A_lpcnmf(:,semiSplit) = A_mid;
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
                
                %--------------------------------------------------------------------------------------
                %Clustering by Label Propagation Constrained Non-negative Matrix Factorization(LpCNMF)
                options = [];
                options.WeightMode = 'HeatKernel';
                options.NeighborMode = 'KNN';
                options.k = 5;
                options.t = 1;
                options.maxIter = 200;
                options.alpha = r;  
                W = constructW(feaSet,options);
                [~, Zest_lpcnmf, F] = LpCNMF(feaSet', A_lpcnmf', k, k, S, W, options); 
                Zest_lpcnmf = F*Zest_lpcnmf;        
                label = litekmeans(Zest_lpcnmf, k, 'Replicates',20); 
                label = bestMap(gndSet,label);
                acc = length(find(gndSet(~semiSplit) == label(~semiSplit)))/length(gndSet(~semiSplit)); 
                nmi = MutualInfo(gndSet(~semiSplit),label(~semiSplit));
                acc_lpcnmf(ClusterIter,PercentIter,RegularIter,runIter) = acc; 
                nmi_lpcnmf(ClusterIter,PercentIter,RegularIter,runIter) = nmi;

                % 打印当前时间
                fp = fopen([aim_root_path,'\','regularization_20_times_6-10classes_log.txt'],'a');
                T = clock; 
                time = [num2str(T(1)),'/',num2str(T(2)),'/',num2str(T(3)),' ',num2str(T(4)),':',num2str(T(5)), ':',num2str(T(6))];
                fprintf(fp,['ClusterIter= ',num2str(ClusterIter)]);
                fprintf(fp,['PercentIter= ',num2str(PercentIter)]);
                fprintf(fp,['RegularIter= ',num2str(RegularIter)]);
                fprintf(fp,['runIter= ',num2str(runIter)]);
                fprintf(fp,[time,'\r\n']);
                fclose(fp);
            end
        end
    end
    
    fprintf('%d   %3.3f   %3.3f   %3.3f    %3.3f   %3.3f   %3.3f   %3.3f   %3.3f   %3.3f \n',...
        k, mean(acc_lpcnmf(ClusterIter,1,4,:)), mean(acc_lpcnmf(ClusterIter,2,4,:)), ...
        mean(acc_lpcnmf(ClusterIter,3,4,:)), mean(acc_lpcnmf(ClusterIter,4,4,:)), ...
        mean(acc_lpcnmf(ClusterIter,5,4,:)), mean(acc_lpcnmf(ClusterIter,6,4,:)), ...
        mean(acc_lpcnmf(ClusterIter,7,4,:)), mean(acc_lpcnmf(ClusterIter,8,4,:)), ...
        mean(acc_lpcnmf(ClusterIter,9,4,:)));

    save([aim_root_path,'\',dataset,' acc_lpcnmf_regularization'],'acc_lpcnmf');
    save([aim_root_path,'\',dataset,' nmi_lpcnmf_regularization'],'nmi_lpcnmf');

end
toc;


