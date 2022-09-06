%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fscore refer to https://en.wikipedia.org/wiki/F-score#cite_note-2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function FMeasure = Fscore(P,C)
    % P true class
    % C predict class
    N = length(C);% sample number
    p = unique(P);
    c = unique(C);
    P_size = length(p);% number of true class
    C_size = length(c);%  number of predict class

    Pid = double(ones(P_size,1)*P == p'*ones(1,N) );
    Cid = double(ones(C_size,1)*C == c'*ones(1,N) );
    CP = Cid*Pid'; % C*P
    Pj = sum(CP,1);
    Ci = sum(CP,2);
    
    precision = CP./( Ci*ones(1,P_size) );
    recall = CP./( ones(C_size,1)*Pj );
    F = 2*precision.*recall./(precision+recall);
    % total F
    FMeasure = sum( (Pj./sum(Pj)).*max(F));
    end