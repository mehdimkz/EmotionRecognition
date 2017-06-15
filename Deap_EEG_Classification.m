tic;
clc
clear all
close all
Name=['D:\Dropbox\PhD Expriments\Matlab\DEAP_MatlabCodes\Deap_EEG_features_subtracing216.mat'];
load(Name);

Tr=1; %classification based on Valence=1,Arousal=2,Liking=4......Must be set before running/ 5 is for classification based on emotional state


%for making progress bar
h = waitbar(0,'Please wait...');
steps=100;
step=0;
%Data=featmatlabels;

nInput=216;  %numbers of totoal features 

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

%normalaized=feature_table; %for testing non normalaized  data
Data=cat(2,normalaized(:,:),labels(:,:))

%***Implementation of One viedo-out- SVM Cross validation

z=32;

for x=1:z %Subjects-seprating each subject
       clearvars Subject_Data TrnData TstData;
    %showing the progress of program
   step=step+1;
   waitbar(step/steps,h,sprintf('%12.9f',step));   

if x==1
 
    Subject_Data=Data(1:40,:);
else
   
    Subject_Data=Data(((x-1)*40+1):(x*40),:);
     
end

%One viedo out
for j=1:40
[TrnData] = removerows(Subject_Data(:,:),'ind',[j]);
TstData=Subject_Data(j,:)  

%Set Target Label and make Train Data and test data table  with their joined target label
TrnData=cat(2,TrnData(:,1:nInput),TrnData(:,nInput+Tr)); %Assign one target labels to observatios..based on Valence Arousal Likinking
TstData=cat(2,TstData(:,1:nInput),TstData(:,nInput+Tr)); %Assign one target labels to observatios..based on Valence Arousal Likinking

%***SVM***
model = svmtrain(TrnData(:,nInput+1),TrnData(:,1:nInput), '-b 1'); %chnage the index for Arousal/valence/liking
predicted_label = svmpredict(TstData(:,nInput+1), TstData(:,1:nInput), model);
SVM_prd_labels((x-1)*40+j,1)=predicted_label;

%***CART***
tree = ClassificationTree.fit(TrnData(:,1:nInput),TrnData(:,nInput+1)); %given by Adel
[CART_predicted, ScoreCARTst]= predict(tree, TstData(:,1:nInput));
CART_prd_labels((x-1)*40+j,1)=CART_predicted;

%***ANN***
 TrnData1=TrnData(:,1:nInput+1);
 TstData1=TstData(:,1:nInput+1);
 N_Class=length(unique(TrnData1(:,size(TrnData1,2))));

 t=zeros(N_Class,size(TrnData1,1));
 
 for ii=1:size(TrnData1,1)
     
     t(TrnData1(ii,size(TrnData1,2)),ii)=1;
 end


%%%%%% ANN  %%%%%%%%%

No_neurons= ceil((size(TrnData1,2)+N_Class)/2);

 net = patternnet(No_neurons);
 
 
 [net,tr] = train(net,TrnData1(:,1:nInput)',t);
 
 
 % Testing 
 KFTstScore = net(TstData1(:,1:nInput)');
 testIndices = vec2ind(KFTstScore);
 KFTstclass=testIndices';  % This gives you the output of testing data

% Training
  KFTrnScore = net(TrnData(:,1:nInput)');
 testIndices1 = vec2ind(KFTrnScore);
 KFTrnclass=testIndices1';  % This gives you the output of training data

ANN_prd_labels((x-1)*40+j,1)=KFTstclass;

%***Random Forest***

Rf_model = classRF_train(TrnData(:,1:nInput),TrnData(:,nInput+1),1000);
Rf_predicted = classRF_predict(TstData(:,1:nInput),Rf_model);
Rf_prd_labels((x-1)*40+j,1)=Rf_predicted;

%***LDA****
 LDA_model = fitcdiscr (TrnData(:,1:nInput),TrnData(:,nInput+1),'discrimType','diagLinear');
 LDA_predicted_label=predict(LDA_model,TstData(:,1:nInput));
 LDA_prd_labels((x-1)*40+j,1)=LDA_predicted_label;
 
 %***Naive Bayes**
 %{
 %"In NB,For Gaussian distribution, each class(low or high) must have at least two
 %observations." So:
 if (sum(double(ismember(TrnData(:,nInput+1),[1])))>1 && sum(double(ismember(TrnData(:,nInput+1),[2])))>1)
 NaiveB_class = fitNaiveBayes(TrnData(:,1:nInput),TrnData(:,nInput+1));
 NaiveB_predicted_label = NaiveB_class.predict(TstData(:,1:nInput));
 NB_prd_labels((x-1)*40+j,1)=NaiveB_predicted_label;
 nb=1;
 else
     nb=0; %means:NB classification can not be used for this set of training data
 end
 %}
end

 Subject_Data(:,nInput+Tr)== SVM_prd_labels((40*x-39):(x*40),1)
 SVM_Accuracy_subjective(x,1)=mean(ans)*100; 
 
 Subject_Data(:,nInput+Tr)== CART_prd_labels((40*x-39):(x*40),1)
 CART_Accuracy_subjective(x,1)=mean(ans)*100; 
 
 Subject_Data(:,nInput+Tr)== ANN_prd_labels((40*x-39):(x*40),1)
 ANN_Accuracy_subjective(x,1)=mean(ans)*100; 
 
 
 Subject_Data(:,nInput+Tr)== Rf_prd_labels((40*x-39):(x*40),1)
 Rf_Accuracy_subjective(x,1)=mean(ans)*100;
 
 Subject_Data(:,nInput+Tr)== LDA_prd_labels((40*x-39):(x*40),1)
 LDA_Accuracy_subjective(x,1)=mean(ans)*100;
 %{
 if nb>0
 Subject_Data(:,nInput+Tr)== NB_prd_labels((40*x-39):(x*40),1)
 NB_Accuracy_subjective(x,1)=mean(ans)*100;
 end
 %}
end

  labels(:,Tr)==SVM_prd_labels(:,1); %chnage label index  for arousal valence/liking
  SVM_final_accuracy=double(mean(ans)*100); 
 
 labels(:,Tr)==CART_prd_labels(:,1); %chnage label index  for arousal valence/liking
 CART_final_accuracy=double(mean(ans)*100); 

 labels(:,Tr)==ANN_prd_labels(:,1); %chnage label index  for arousal valence/liking
 ANN_final_accuracy=double(mean(ans)*100); 
 
 labels(:,Tr)==Rf_prd_labels(:,1); %chnage label index  for arousal valence/liking
 Rf_final_accuracy=double(mean(ans)*100); 

 labels(:,Tr)==LDA_prd_labels(:,1); %chnage label index  for arousal valence/liking
 LDA_final_accuracy=double(mean(ans)*100); 

%{ 
 if nb>0
 labels(:,Tr)==NB_prd_labels(:,1); %chnage label index  for arousal valence/liking
 NB_final_accuracy=double(mean(ans)*100); 
 end
 %}
 toc;