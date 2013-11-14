clear;

methods = {'quad-rank-100-5.0-1.0', 'quad-omm-100-5.0-1.0', 'quad-mle-100-5.0'};

allAcc = {};
allROC = {};
allPR = {};

%% read prediction output

fin = fopen('../savedResults/results-50nodes.txt', 'r');
while ~feof(fin)
    line = fgetl(fin);
    pattern = 'Method ([^,]+), fold ([^,]+), auprc positive: ([^,]+), negative: ([^,]+), auROC: ([^,]+), rounded accuracy: ([^,]+)';
    %Method quad-mle-100-5.0, fold 0, auprc positive: 0.364541623978603, negative: 0.9790345053226167, auROC: 0.8168291592832011, rounded accuracy: 0.778
    matches = regexp(line, pattern, 'tokens');
    if ~isempty(matches)
        matches = matches{1};
        method = matches{1};
        fold = str2num(matches{2})+1;
        aupr = str2num(matches{3});
        auprneg = str2num(matches{4});
        auroc = str2num(matches{5});
        acc = str2num(matches{6});
        
        i = find(strcmp(method, methods));
        
        if ~isempty(i)
            allAcc{i}(fold) = acc;
            allROC{i}(fold) = auroc;
            allPR{i}(fold) = aupr;
        end
    end
end

%% grab latest results

fold = fold + 1;
i = 1;

fin = fopen('../results.tmp.txt', 'r');
while ~feof(fin)
    line = fgetl(fin);
    pattern = 'Area under positive-class PR curve: (.+)';
    matches = regexp(line, pattern, 'tokens');
    if ~isempty(matches)
        allPR{i}(fold) = str2double(matches{1}{1});
    end
    pattern = 'Area under ROC curve: (.+)';
    matches = regexp(line, pattern, 'tokens');
    if ~isempty(matches)
        allROC{i}(fold) = str2double(matches{1}{1});
    end
    pattern = 'Rounded accuracy: (.+)';
    matches = regexp(line, pattern, 'tokens');
    if ~isempty(matches)
        allAcc{i}(fold) = str2double(matches{1}{1});
    end
    pattern = 'Method ([^,]+), fold ([^,]+), acc ([^,]+)';
    matches = regexp(line, pattern, 'tokens');
    if ~isempty(matches)
        method = matches{1}{1};
        j = find(strcmp(method, methods));
        assert(i == j);
        
        assert(str2num(matches{1}{2}) == fold - 10 - 1);
        
        i = i + 1;
        if (i == 4)
            fold = fold + 1;
            i = 1;
        end
    end
end

% %% truncate

totalRuns = 30;

for i = 1:length(methods)
  allAcc{i} = allAcc{i}(10:totalRuns);
  allROC{i} = allROC{i}(10:totalRuns);
  allPR{i} = allPR{i}(10:totalRuns);  
end

%% compute means
for i = 1:length(methods)
    meanAcc(i) = mean(allAcc{i});
    meanROC(i) = mean(allROC{i});
    meanPR(i) = mean(allPR{i});
end


%% get best method

[~, bestAcc] = max(meanAcc);
[~, bestROC] = max(meanROC);
[~, bestPR] = max(meanPR);

%% sig test all vs best

threshold = 0.05;
for i = 1:length(methods)
%     sigAcc(i) = ttest2(allAcc{bestAcc}, allAcc{i}, threshold);
%     sigROC(i) = ttest2(allROC{bestROC}, allROC{i}, threshold);
%     sigPR(i) = ttest2(allPR{bestPR}, allPR{i}, threshold);
    sigAcc(i) = (ttest(allAcc{bestAcc}, allAcc{i}, threshold));
    sigROC(i) = (ttest(allROC{bestROC}, allROC{i}, threshold));
    sigPR(i) = (ttest(allPR{bestPR}, allPR{i},  threshold));
end

% Here precision is multiclass classification accuracy

%% print latex
latexNames = {'pROC', 'L1', 'None'};

fprintf('BEGIN RANKING RESULTS TABLE\n\n')
fprintf('\\begin{tabular}{lrrr}\n')
fprintf('\\toprule\n')
fprintf('  & ROC & P-R & Acc.\\\\\n')
fprintf('\\midrule\n')

for i = 1:length(latexNames)
    fprintf(latexNames{i});
    if i ~= bestROC && sigROC(i)
        fprintf('& %0.3f (%0.3f)', meanROC(i), std(allROC{i}));
    else
        fprintf('& \\textbf{%0.3f (%0.3f)}', meanROC(i), std(allROC{i}));
    end
    if i ~= bestPR && sigPR(i)
        fprintf('& %0.3f (%0.3f)', meanPR(i), std(allPR{i}));
    else
        fprintf('& \\textbf{%0.3f (%0.3f)}', meanPR(i), std(allPR{i}));
    end
    if i ~= bestAcc && sigAcc(i)
        fprintf('& %0.3f (%0.3f)', meanAcc(i), std(allAcc{i}));
    else
        fprintf('& \\textbf{%0.3f (%0.3f)}', meanAcc(i), std(allAcc{i}));
    end
    fprintf('\\\\\n');
end

fprintf('\\bottomrule\n')
fprintf('\\end{tabular}\n')
fprintf('\nEND RANKING RESULTS TABLE\n')

