
%Applying Feature selection on training data... and choose the same
%features for testing data...predicting the lables using diffrent set of
%features ranked by feature selection (e.g 18,36,54,...)...
% Prodcing lables coulmn by coulmn by row (produce all labels for each testing case
% usuing diffrent set of features.
clear all
Name=['D:\Dropbox\PhD Expriments\Databases\Sc&EMG&RSP&ECG_Humain.mat'];
load(Name);

m=100;
z=m;
L=1;    %have to set target label(attribute)coloumn for classification ,1 for emotion classiofcation,2 for Arousal classification ,3 for Valence classification
tic; 
f=8; %***numbers of feature selection algorithem-must be set

%Make label for Arousal and Valence High =2 Low=1 based on 2 dimension
%mapping of emotional state

for j=1:100
    
   switch labels(j,1)
    case 1
        labels(j,2)=2      %Arousal Label for joy
        labels(j,3)=2      %Valence Label  for joy
    case 2
        labels(j,2)=2      %Arousal Label for anger
        labels(j,3)=1      %Valence Label  for anger
    case 3
        labels(j,2)=1      %Arousal Label for sadness
        labels(j,3)=1      %Valence Label  for sadness
    case 4
        labels(j,2)=1      %Arousal Label for Pleasure
        labels(j,3)=2      %Valence Label  for Pleasure
end 
end

%**make matrix ready for feature selection process

%nInput: number of selected features,
% Trninput: training data,
%Trnoutput: output labels
nInput=186;  %numbers of totoal features that should be ranked.
Data=cat(2,featmat(1:100,:),labels(1:100,1));
    
for i=1:f %f is numbers of featureselection algorithem
    
    clearvars newfeatures features  selectedFeatures training_set training_instance_label_vector testing_set testing_set_label;


 
x=10;%number of feature set 18,36,54..max is 10
d=18;%number of features of  first feature set

for n=1:x
%%Take one record out as a testing the rest of data as
%trainingset...repeating for all records


for j=1:100
clearvars selectedFeatures training_newfeatmat testing_newfeatmat;   
[TrnData] = removerows(Data(:,:),'ind',[j]);  %one leave out
TstData=Data(j,:)


 if i==1
 
  methodFS='jmi';

 elseif i==2
  methodFS='cmim';
 else
   
  if i==3 
   methodFS='disr';
   elseif i==4 
   methodFS='mim';
   else
   if i==5 
   methodFS='cife';
   elseif i==6 
   methodFS='icap';
   else
   if i==7 
   methodFS='condred';
   elseif i==8 
   methodFS='relief';
   end
   %if n==9 
   %methodFS='cmi';  has problem select only 2 features
   %end
   end
   end
     
   end
     
   [selectedFeatures]=feast(methodFS,nInput,TrnData(:,1:nInput),TrnData(:,nInput+1));
 %}

%{

if i==1
 methodFS='relief';
 [selectedFeatures]=feast(methodFS,nInput,TrnData(:,1:nInput), TrnData(:,nInput+1)); 
end
%
if i==2
 methodFS='jmi';
 [selectedFeatures]=feast(methodFS,nInput,TrnData(:,1:nInput), TrnData(:,nInput+1)); 
end 

if i==3
 methodFS='cife';
 [selectedFeatures]=feast(methodFS,nInput,TrnData(:,1:nInput), TrnData(:,nInput+1)); 
end 



if i==4
    
   fisherfeatures=fsFisher(TrnData(:,1:nInput),TrnData(:,nInput+1)); % fisher feature ranking-Feature Selection Package - Algorithms - Fisher Score 
   selectedFeatures=fisherfeatures.fList'; 
end
%}
%{
if i==3 %SVM feature ranking....Produced by weka  saved in text file
     
    tmp_data = importdata('svmfst.txt',',');
     selectedFeatures=tmp_data';
end



if i==5
 methodFS='icap';
 [selectedFeatures]=feast(methodFS,nInput,TrnData(:,1:nInput),TrnData(:,nInput+1)); 
end


if i==6 %t-Test feature ranking
   Ttestfeatures=fsTtest(TrnData(:,1:nInput),TrnData(:,nInput+1)); % fisher feature ranking-Feature Selection Package - Algorithms - Fisher Score 
  selectedFeatures=Ttestfeatures.fList;
  end

%}
training_newfeatmat=[TrnData(:,selectedFeatures(1:d))];
testing_newfeatmat=[TstData(:,selectedFeatures(1:d))];

%****CART
%{
tree = ClassificationTree.fit(training_newfeatmat,TrnData(:,nInput+1)); %given by Adel
[CART_predicted, ScoreCARTst]= predict(tree,testing_newfeatmat);

CART_predicted_labels(j,10*i-10+n)=CART_predicted;
%}
%NaiveB_class = fitNaiveBayes(training_set,training_instance_label_vector);
%NaiveB_predicted = NaiveB_class.predict(testing_set);


%SVMmodel = svmtrain(training_instance_label_vector, training_set, '-b 1');

%SVM_predicted = svmpredict(testing_set_label, testing_set, SVMmodel);



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


ANN_predicted_labels(j,10*i-10+n)=KFTstclass;

%}
%%%LDA

LDA_model = fitcdiscr (training_newfeatmat,TrnData(:,nInput+1),'discrimType','diagLinear');
LDA_predicted_label=predict(LDA_model,testing_newfeatmat);

LDA_predicted_labels(j,10*i-10+n)=LDA_predicted_label;

%}




%**Random Forest Modeling

%Rf_model = classRF_train(training_set,training_instance_label_vector,1000);
%Rf_predicted = classRF_predict(testing_set,Rf_model);
%{
%***Ensemble Method


 %Mdl = fitensemble(training_set,training_instance_label_vector,'AdaBoostM1',100,'Tree')
 %Mdl = fitensemble(training_set,training_instance_label_vector,'Subspace',100,'Discriminant')
  Mdl = fitensemble(training_set,training_instance_label_vector,'Bag',150,'Tree','type','classification')

Ensmb_Predicted = predict(Mdl,testing_set)
%}
%***LDA******



%****calculation of  testing accuracy for each viedo

%testing_set_label==Rf_predicted
%Rf_Accuracy_testing(j,(10*n-10+k)) = (nnz(ans))*100;
%Rf_predicted_labels(j,(10*n-10+k))=Rf_predicted;




%testing_set_label==NaiveB_predicted
%NaiveB_Accuracy_testing(j) = (nnz(ans))*100;
%NaiveB_predicted_Matrix(j)=NaiveB_predicted;

%{
testing_set_label==SVM_predicted
SVM_Accuracy_testing(j,1) = (nnz(ans))*100;
%SVM_predicted_Matrix(j)=SVM_predicted;
%}



%{
testing_set_label==Ensmb_Predicted
Ensmb_Accuracy_testing(j,1) = (nnz(ans))*100;

%}
%{
testing_set_label==LDA_predicted_label
LDA_Accuracy_testing(j,(10*n-10+k)) = (nnz(ans))*100;
LDA_predicted_labels(j,(10*n-10+k))=LDA_predicted_label;
%}
%Mixed models prediction-Majority votes
%testing_set_label==mode(MixModels_predicted)
%Accuracy_testing(x,j) = (nnz(ans))*100;

end

d=d+18;
end 
end 


for i=1:f %f numbers of feature ranking algorithems
for n=1:x  %x numbers of feature sets
    
    labels(:,1)==LDA_predicted_labels(:,10*i-10+n);
    FeatureSets_Accuracy(i,n)=mean(ans)*100; 
end
Majority_lables_rankingalgo(:,i)=mode(LDA_predicted_labels(:,n*i-9:n*i),2);
labels(:,1)== Majority_lables_rankingalgo(:,i);
Featureselection_Accuracy(1,i)=mean(ans)*100;
end
%calculating the majority vote 
% FinalPredicatedLabelsANN = mode(ANN_predicted_labels,2);
 %FinalPredicatedLabelsRF = mode(Rf_predicted_labels,2);
 %FinalPredicatedLabelsANN = mode(ANN_predicted_labels,2);
 FinalPredicatedLabelsLDA = mode(LDA_predicted_labels,2);

labels(:,1)== FinalPredicatedLabelsLDA;
FinalAccuracy=mean(ans)*100;
toc
