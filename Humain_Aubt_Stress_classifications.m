%**Implemnetation   one video(session) leave out CART&SVM&RF&ANN cross validation
%for Humanin dataset-Ausburg toolkit
%load AubT_Stress_EMG&RSP.mat
m=129;
z=m;



for j=1:129
   

if j==1
 
   training_set=featmat((j+1):129,:);
   training_instance_label_vector =labels((j+1):129,:);
else
    if j<100
   training_set=cat(1,featmat(1:(j-1),:),featmat((j+1):129,:));
   training_instance_label_vector =cat(1,labels(1:(j-1),:),labels((j+1):129,:));
    
  else
    training_set=featmat(1:(j-1),:); 
    training_instance_label_vector =labels(1:(j-1),:);
end
end

testing_set=featmat(j,:);
testing_set_label=labels(j,1);

tree = ClassificationTree.fit(training_set,training_instance_label_vector); %given by Adel
[CART_predicted, ScoreCARTst]= predict(tree,testing_set);

%NaiveB_class = fitNaiveBayes(training_set,training_instance_label_vector);
%NaiveB_predicted = NaiveB_class.predict(testing_set);


SVMmodel = svmtrain(training_instance_label_vector, training_set, '-b 1');

SVM_predicted = svmpredict(testing_set_label, testing_set, SVMmodel);

%Making ANN model

nInput1=64;
TrnData=cat(2,training_set,training_instance_label_vector);
TstData=cat(2,testing_set,testing_set_label);

N_Class=length(unique(TrnData(:,size(TrnData,2))));

 t=zeros(N_Class,size(TrnData,1));
 
 for ii=1:size(TrnData,1)
     
     t(TrnData(ii,size(TrnData,2)),ii)=1;
 end


%%%%%% ANN  %%%%%%%%%

No_neurons= ceil((size(TrnData,2)+N_Class)/2);

 net = patternnet(No_neurons);
 
 
 [net,tr] = train(net,TrnData(:,1:nInput1)',t);
 
 
 % Testing 
 KFTstScore = net(TstData(:,1:nInput1)');
 testIndices = vec2ind(KFTstScore);
 KFTstclass=testIndices';  % This gives you the output of testing data

% Training
  KFTrnScore = net(TrnData(:,1:nInput1)');
 testIndices1 = vec2ind(KFTrnScore);
 KFTrnclass=testIndices1';  % This gives you the output of training data
%} 


%**Random Forest Modeling

Rf_model = classRF_train(training_set,training_instance_label_vector,1000);
Rf_predicted = classRF_predict(testing_set,Rf_model);

%***Ensemble Method


 %Mdl = fitensemble(training_set,training_instance_label_vector,'AdaBoostM1',100,'Tree')
 %Mdl = fitensemble(training_set,training_instance_label_vector,'Subspace',100,'Discriminant')
 Mdl = fitensemble(training_set,training_instance_label_vector,'Bag',150,'Tree','type','classification')

Ensmb_Predicted = predict(Mdl,testing_set)


%LDA requires enough information to be able to estimate a full-rank covariance matrix, and at 
%a minimum that means more observations(records) than variables (attributes). You might think 
%about selecting a good subset of variables somehow, or constructing new 
%variables using, for example, PCA. 

%LDA_pridicted_labels = classify(testing_set,training_set,training_instance_label_vector);


%make matrix for prediction of each model
%MixModels_predicted=cat(1,SVM_predicted,Rf_predicted) 


%****calculation of  testing accuracy for each viedo
testing_set_label==Rf_predicted
RF_Accuracy_testing(j,1) = (nnz(ans))*100;

testing_set_label==CART_predicted
CART_Accuracy_testing(j,1) = (nnz(ans))*100;
%CART_predicted_Matrix(j)=CART_predicted;   %to make labels matrix to finally used for second classification

 %testing_set_label==NaiveB_predicted
 %NaiveB_Accuracy_testing(j) = (nnz(ans))*100;
%NaiveB_predicted_Matrix(j)=NaiveB_predicted;

testing_set_label==SVM_predicted
SVM_Accuracy_testing(j,1) = (nnz(ans))*100;
%SVM_predicted_Matrix(j)=SVM_predicted;

testing_set_label==KFTstclass
ANN_Accuracy_testing(j,1) = (nnz(ans))*100;




testing_set_label==Ensmb_Predicted
Ensmb_Accuracy_testing(j,1) = (nnz(ans))*100;


%testing_set_label==LDA_pridicted_labels
%LDA_Accuracy_testing(j,1) = (nnz(ans))*100;

%Mixed models prediction-Majority votes
%testing_set_label==mode(MixModels_predicted)
%Accuracy_testing(x,j) = (nnz(ans))*100;

end

%Make overall accuracy for each participants  as tested for 40 viedos left
%out
CART_accuracy=mean(CART_Accuracy_testing(:,1));
%NaiveB_accuracy=mean(NaiveB_Accuracy_testing(:,1));
SVM_accuracy=mean(SVM_Accuracy_testing(:,1));
ANN_accuracy=mean(ANN_Accuracy_testing(:,1));
RF_accuracy=mean(RF_Accuracy_testing(:,1));
Ensemb_accuracy=mean(Ensmb_Accuracy_testing(:,1));
%LDA_accuracy=mean(LDA_Accuracy_testing(:,1));

