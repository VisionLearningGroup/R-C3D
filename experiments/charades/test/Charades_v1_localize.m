function [rec_all,prec_all,ap_all,map]=Charades_v1_localize(clsfilename,gtpath)
%
%     Input:    clsfilename: path of the input file
%                    gtpath: the path of the groundtruth file
%
%    Output:        rec_all: recall
%                  prec_all: precision
%                    ap_all: AP for each class
%                       map: MAP 
%
% Please refer to the README.txt file for an overview of how localization performance is evaluated
% 
% Example:
%
%  [rec_all,prec_all,ap_all,map]=Charades_v1_localize('test_submission.txt','Charades_v1_test.csv');
%
% Code adapted from THUMOS15 
%

tic;
fprintf('Loading Charades Annotations:\n');
frames_per_video = 25;
[gtids,gtclasses] = load_charades_localized(gtpath,frames_per_video);
nclasses = 157;
ntest = length(gtids);
toc; tic;

% load test scores
fprintf('Reading Submission File:\n');
[testids,framenr,testscores]=textread(clsfilename,'%s%d%[^\n]');
if min(framenr)==0 
    fprintf('Warning: Frames should be 1 indexed\n');
    fprintf('Warning: Adding 1 to all frames numbers\n');
    framenr = framenr+1;
end
toc; tic;
fprintf('Parsing Submission Scores:\n'); 
nInputNum=size(testscores,1);
if nInputNum<ntest
    fprintf('Warning: %d Total frames missing\n',ntest-nInputNum);
end
testscoresparsed = cellfun(@str2num,testscores,'UniformOutput',false);
eleNum=length(testscoresparsed{1});
if eleNum~=nclasses&&eleNum~=nclasses+1
    fprintf('Error: Incompatible number of classes\n');
end
make_frameid = @(x,y) [x,'-',sprintf('%03d',y)];
frameids = cellfun(make_frameid,testids,num2cell(framenr),'UniformOutput',false);
predictions = containers.Map(frameids,testscoresparsed);
toc; tic;

% compare test scores to ground truth
fprintf('Constructing Ground Truth Matrix:\n')
gtlabel = zeros(ntest,nclasses);
test = -inf(ntest,nclasses);
for i=1:ntest
    id = gtids{i};
    gtlabel(i,gtclasses{i}+1) = 1;
    if predictions.isKey(id)
        test(i,:) = predictions(id);
    end
end
toc; tic;

for i=1:nclasses
    [rec_all(:,i),prec_all(:,i),ap_all(:,i)]=THUMOSeventclspr(test(:,i),gtlabel(:,i));
end
map=mean(ap_all);
wap=sum(ap_all.*sum(gtlabel,1))/sum(gtlabel(:));
fprintf('\n\n')
fprintf('Per-Frame MAP: %f\n',map);
fprintf('Per-Frame WAP: %f (weighted by size of each class)',wap);
fprintf('\n\n')


function [rec,prec,ap]=THUMOSeventclspr(conf,labels)
[so,sortind]=sort(-conf);
tp=labels(sortind)==1;
fp=labels(sortind)~=1;
npos=length(find(labels==1));

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);

% compute average precision

ap=0;
tmp=labels(sortind)==1;
for i=1:length(conf)
    if tmp(i)==1
        ap=ap+prec(i);
    end
end
ap=ap/npos;


function [gtids,gtclasses] = load_charades_localized(gtpath,frames_per_video)
% Loads the ground truth annotations from the csv file
f = fopen(gtpath);

% read column headers
headerline = textscan(f,'%s',1);
headerline = regexp(headerline{1}{1},',','split');
ncols = length(headerline);
headers = struct();
for i=1:ncols
    headers = setfield(headers,headerline{i},i);
end

% read data
gtcsv = textscan(f,repmat('%q ',[1 ncols]),'Delimiter',',');
fclose(f);
ntest = size(gtcsv{1},1);
framechar = char(cellfun(@(x) sprintf('%03d',x),num2cell((1:50)'),'UniformOutput',false')); %for speed
gtids = cell(frames_per_video*ntest,1);
gtclasses = cell(frames_per_video*ntest,1);
uncell = @(x) x{1};
c = 1;
for i=1:ntest
    id = gtcsv{headers.id}{i};
    classes = gtcsv{headers.actions}{i};
    time = str2double(gtcsv{headers.length}{i});
    if strcmp(classes,'') 
        missing = true;
    else
        missing = false;
        classes = regexp(classes,';','split')';
        classes = cellfun(@(x) uncell(textscan(x,'c%f %f %f','CollectOutput',true)), classes,'UniformOutput',false); %for speed
        classes = cell2mat(classes);
        % classes is C by 3 matrix where each row is [class start end] for an action
    end
    for j=1:frames_per_video
        frameclasses = zeros(50,1); %for speed
        fc = 1;
        timepoint = (j-1)/frames_per_video*time;
        for k=1:size(classes,1)
            if missing; continue; end
            if (classes(k,2) <= timepoint) && (timepoint <= classes(k,3))
                frameclasses(fc) = classes(k,1);
                fc = fc+1;
            end
        end
        frameid = [id,'-',framechar(j,:)]; %for speed
        gtids{c} = frameid;
        gtclasses{c} = frameclasses(1:(fc-1));
        c = c+1;
    end
end
gtids = gtids(1:c-1);
gtclasses = gtclasses(1:c-1);


