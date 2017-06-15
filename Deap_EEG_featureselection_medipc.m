tic;
clc
clear all
close all
Name=['D:\Dropbox\PhD Expriments\Matlab\DEAP_MatlabCodes\Deap_EEG_features_subtracing216.mat'];
load(Name);

Tr=2; %classification based on Valence=1,Arousal=2,Liking=4,Emotional State=5......Must be set before running

%Make emotinal labels for data based on arousal and valence
for i=1:size(labels,1)
    
  if labels(i,1)==2 & labels(i,2)==2
 
    labels(i,5)=1; %Joy  

 elseif labels(i,1)==1 & labels(i,2)==2
  
    labels(i,5)=2; %Anger  

 else
   
  if labels(i,1)==1 & labels(i,2)==1 
       labels(i,5)=3; %Sadness 
       
  elseif labels(i,1)==2 & labels(i,2)==1  
       
   labels(i,5)=4; %Pleasure 
  
  end 
  end
end


%for making progress bar
h = waitbar(0,'Please wait...');
steps=100;
step=0;

nInput=216;  %numbers of total features 
%Data=featmatlabels;
z=32; %number of subjcts
%***Implementation of One viedo-out- SVM Cross validation
f=10; %determine number of feature ranking methods

%Data Normalization
%{

http://www.mathworks.com/matlabcentral/answers/75568-how-can-i-normalize-data-between-0-and-1-i-want-to-use-logsig
http://stackoverflow.com/questions/10364575/normalization-in-variable-range-x-y-in-matlab
If you want to normalize to [x, y], first normalize to [0, 1] via:

 range = max(a) - min(a);
 a = (a - min(a)) / range;

in case of having mines values in matrix use below code for randomize
between 0-1
NDATA = mat2gray(DATA);

Then scale to [x,y] via:

 range2 = y - x;
 a = (a*range2) + x;

%}


%first normalaize to 0&1
for i = 1:nInput
D(:,i) = mat2gray(feature_table(:,i))
end

%second normalaized data between 1&5/(5-1)=4
normalaized= (D*4) + 1;

Data=cat(2,normalaized(:,:),labels(:,:))

%Data=featmatlabels;


for i=1:f %f is numbers of featureselection algorithem
    
    clearvars newfeatures features  selectedFeatures training_set training_instance_label_vector testing_set testing_set_label;


 
x=9;%number of feature set 18,36,54..max is 10
d=24;%number of features of  each feature set

for n=1:x
  %showing the progress of program
    step=step+1;
   waitbar(step/steps,h,sprintf('%12.9f',step));  
for s=1:z %Subjects-seprating each subject
    
   

if s==1
 
    Subject_Data=Data(1:40,:);
else
   
    Subject_Data=Data(((s-1)*40+1):(s*40),:);
     
end

%One viedo out
for j=1:40
[TrnData] = removerows(Subject_Data(:,:),'ind',[j]);
TstData=Subject_Data(j,:)  
%%Set Target Label and make Train Data and test data table  with their joined target label
TrnData=cat(2,TrnData(:,1:nInput),TrnData(:,nInput+Tr)); %Assign one target labels to observatios..based on Valence Arousal Likinking
TstData=cat(2,TstData(:,1:nInput),TstData(:,nInput+Tr)); %Assign one target labels to observatios..based on Valence Arousal Likinking


if i==1
 
  methodFS='jmi';
  feastalgorithem(i,1)=cellstr('jmi'); % save the name of feature-selection algorthimcc

 elseif i==2
  methodFS='cmim';
  feastalgorithem(i,1)=cellstr('cmim');

 else
   
  if i==3 
   methodFS='disr';
   feastalgorithem(i,1)=cellstr('disr');
   elseif i==4 
   methodFS='mim';
   feastalgorithem(i,1)=cellstr('mim');
  else
   if i==5 
   methodFS='cife';
   feastalgorithem(i,1)=cellstr('cife');
   elseif i==6 
   methodFS='icap';
   feastalgorithem(i,1)=cellstr('icap');

   else
   if i==7 
   methodFS='condred';
   feastalgorithem(i,1)=cellstr('condred');

   elseif i==8 
   methodFS='relief';
   feastalgorithem(i,1)=cellstr('relief');

   end
   
   end
   end
     
end
 
  
   [selectedFeatures]=feast(methodFS,nInput,TrnData(:,1:nInput),TrnData(:,nInput+1));
 
  %}
%{
 if i==1
   methodFS='jmi';
  [selectedFeatures]=feast(methodFS,nInput,TrnData(:,1:nInput), TrnData(:,nInput+1)); 
 end
 if i==2
   methodFS='mim';
  [selectedFeatures]=feast(methodFS,nInput,TrnData(:,1:nInput), TrnData(:,nInput+1)); 
  end
 if i==3
  
  methodFS='icap';
  [selectedFeatures]=feast(methodFS,nInput,TrnData(:,1:nInput), TrnData(:,nInput+1)); 
   
 end
 
 if i==4
  
  methodFS='condred';
  [selectedFeatures]=feast(methodFS,nInput,TrnData(:,1:nInput), TrnData(:,nInput+1)); 
   
 end
%}
  
%{
 if i==1  %use this condition only once Tr=4(liking) otherwise use it with other feast methods
  
  methodFS='relief';
  
  if (s~=1 || j~=32 && j~=36 ) % viedos 32&36 of subject 1 having problem with relief
  
 [selectedFeatures]=feast(methodFS,nInput,TrnData(:,1:nInput), TrnData(:,nInput+1)); 
  end
  feastalgorithem(i,1)=cellstr('relief');
 end
 %}
 
  
 if i==9
  fisherfeatures=fsFisher(TrnData(:,1:nInput),TrnData(:,nInput+1)); % fisher feature ranking-Feature Selection Package - Algorithms - Fisher Score 
   selectedFeatures=fisherfeatures.fList';
   feastalgorithem(i,1)=cellstr('fisher');
 end
 
 

  if i==10
   Ttestfeatures=fsTtest(TrnData(:,1:nInput),TrnData(:,nInput+1)); % fisher feature ranking-Feature Selection Package - Algorithms - Fisher Score 
   selectedFeatures=Ttestfeatures.fList; 
   feastalgorithem(i,1)=cellstr('T-test');
  end
  %}
 
   training_newfeatmat=[TrnData(:,selectedFeatures(1:d))];
   testing_newfeatmat=[TstData(:,selectedFeatures(1:d))];
  
   
   %****CART

   tree = ClassificationTree.fit(training_newfeatmat,TrnData(:,nInput+1)); %given by Adel
   [CART_predicted, ScoreCARTst]= predict(tree,testing_newfeatmat);
   CART_prd_labels((s-1)*40+j,9*i-9+n)=CART_predicted;
   
  %{ 
   %***SVM***
   
   model = svmtrain(TrnData(:,nInput+1),training_newfeatmat, '-b 1'); %chnage the index for Arousal/valence/liking
   predicted_label = svmpredict(TstData(:,nInput+1), testing_newfeatmat, model);
   SVM_prd_labels((s-1)*40+j,9*i-9+n)=predicted_label;
   
   
   %***LDA****
   LDA_model = fitcdiscr (training_newfeatmat,TrnData(:,nInput+1),'discrimType','diagLinear');
   LDA_predicted_label=predict(LDA_model,testing_newfeatmat);
   LDA_prd_labels((s-1)*40+j,9*i-9+n)=LDA_predicted_label;
  
   %}
   %*****Making ANN model
%{
   nInput1=d;
   Trnda=cat(2,training_newfeatmat,TrnData(:,nInput+1));
   Tstda=cat(2,testing_newfeatmat,TstData(:,nInput+1));

   N_Class=length(unique(Trnda(:,size(Trnda,2))));

   t=zeros(N_Class,size(Trnda,1));
 
   for ii=1:size(Trnda,1)
     
     t(Trnda(ii,size(Trnda,2)),ii)=1;
   end


  %%%%%% ANN  %%%%%%%%%

  No_neurons= ceil((size(Trnda,2)+N_Class)/2);

  net = patternnet(No_neurons);
 
 
  [net,tr] = train(net,Trnda(:,1:nInput1)',t);
 
 
  % Testing 
  KFTstScore = net(Tstda(:,1:nInput1)');
  testIndices = vec2ind(KFTstScore);
  KFTstclass=testIndices';  % This gives you the output of testing data


ANN_prd_labels((s-1)*40+j,9*i-9+n)=KFTstclass;
   
%}

end
    
      Subject_Data(:,nInput+Tr)== CART_prd_labels((40*s-39):(s*40),9*i-9+n)
      CART_Accuracy_subjective(s,9*i-9+n)=double(mean(ans)*100);
  %{
      Subject_Data(:,nInput+Tr)== SVM_prd_labels((40*s-39):(s*40),9*i-9+n)
      SVM_Accuracy_subjective(s,9*i-9+n)=double(mean(ans)*100); 
     

      Subject_Data(:,nInput+Tr)== LDA_prd_labels((40*s-39):(s*40),9*i-9+n)
      LDA_Accuracy_subjective(s,9*i-9+n)=double(mean(ans)*100); 
%}
    %}
   %{ 
    Subject_Data(:,nInput+Tr)== ANN_prd_labels((40*s-39):(s*40),9*i-9+n)
    ANN_Accuracy_subjective(s,9*i-9+n)=double(mean(ans)*100); 
    
    %}
end 
   
    labels(:,Tr)==CART_prd_labels(:,9*i-9+n); %chnage label index  for arousal valence/liking
    CART_accuracy_featureset(i,n)=double(mean(ans)*100);
%{
    labels(:,Tr)==SVM_prd_labels(:,9*i-9+n); %chnage label index  for arousal valence/liking
    SVM_accuracy_featureset(i,n)=double(mean(ans)*100);
   

    labels(:,Tr)==LDA_prd_labels(:,9*i-9+n); %chnage label index  for arousal valence/liking
    LDA_accuracy_featureset(i,n)=double(mean(ans)*100);
  %} 
%{
    labels(:,Tr)==ANN_prd_labels(:,9*i-9+n); %chnage label index  for arousal valence/liking
    ANN_accuracy_featureset(i,n)=double(mean(ans)*100);
    %}
    d=d+24;
end



CART_Majority_lables_rankingalgo(:,i)=mode(CART_prd_labels(:,n*i-8:n*i),2);
labels(:,Tr)== CART_Majority_lables_rankingalgo(:,i);
CART_Accuracy_rankingalgo(1,i)=mean(ans)*100;
%{
SVM_Majority_lables_rankingalgo(:,i)=mode(SVM_prd_labels(:,n*i-8:n*i),2);
labels(:,Tr)== SVM_Majority_lables_rankingalgo(:,i);
SVM_Accuracy_rankingalgo(1,i)=mean(ans)*100;


LDA_Majority_lables_rankingalgo(:,i)=mode(LDA_prd_labels(:,n*i-8:n*i),2);
labels(:,Tr)== LDA_Majority_lables_rankingalgo(:,i);
LDA_Accuracy_rankingalgo(1,i)=mean(ans)*100;
%}

%{
Majority_lables_rankingalgo(:,i)=mode(ANN_prd_labels(:,n*i-8:n*i),2);
labels(:,Tr)== Majority_lables_rankingalgo(:,i);
ANN_Accuracy_rankingalgo(1,i)=mean(ans)*100;
%}


end

labels(:,Tr)== mode(CART_prd_labels,2);
MajorityCART_FinalAccuracy=mean(ans)*100;

%{
labels(:,Tr)== mode(SVM_prd_labels,2);
MajoritySVM_FinalAccuracy=mean(ans)*100;


labels(:,Tr)== mode(LDA_prd_labels,2);
MajorityLDA_FinalAccuracy=mean(ans)*100;
%}

%{
labels(:,Tr)== mode(ANN_prd_labels,2);
MajorityANN_FinalAccuracy=mean(ans)*100;
%}
toc;