clc
clear all
close all
%for making progress bar
h = waitbar(0,'Please wait...');
steps=100;
step=0;

for m=1:2  %Subjects


    
if m<10
loadfile=strcat('D:\emotion databses\data_preprocessed_matlab_DEAP\s0',int2str(m),'.mat'); 
else
loadfile=strcat('D:\emotion databses\data_preprocessed_matlab_DEAP\s',int2str(m),'.mat'); 
end

DEAP = load(loadfile);
for n=1:40
    for nn=1:32 %EEG channels
        
        R1(1,:)=DEAP.data(n,nn,:);
        
        % plot(R1)
        
        %%
        fs=128;   % Sample rate
        L=8;    % Order of Filter
        fl=1;    % low cut_off frequency
        fh1=60;   % high cut_off frequency
        L2=20;
        f2=48;
        fh2=52;
        d1 = fdesign.bandpass('N,F3dB1,F3dB2',L,fl,fh1,fs);
        Hd1 = design(d1,'butter');
        filted_data = filter(Hd1,R1);    %bandpass filter (EEG frequency rang)
        d2 = fdesign.bandstop('N,F3dB1,F3dB2',L2,f2,fh2,fs);
        Hd2 = design(d2,'butter');
        EEG_data=filter(Hd2, filted_data);   %bandstop filter (notch)
        
      %  hold on, plot(EEG_data,'r')
        
        %%
        
        
        wlen=ceil(fs*1.024);
        nfft=wlen;
        h=ceil(0.75*wlen);
        [stft1, f, t] = stft(R1, wlen, h, nfft, fs);
        a_stft1=abs(stft1);
        %         figure, surf(t,f,a_stft1)
        dd_stft1(n,nn,:,:)=a_stft1(:,:);
        
        
        %%
        
        s_theta=5;
        e_theta=9;
        psd_theta1(n,nn)=sum(sum (dd_stft1(n,nn,s_theta:e_theta,:))/(f(e_theta)-f(s_theta)));
        
        
        s_alpha=9;
        e_alpha=13;
        psd_alpha1(n,nn)=sum(sum (dd_stft1(n,nn,s_alpha:e_alpha,:))/(f(e_alpha)-f(s_alpha)));
        
        
        s_alpha_slow=9;
        e_alpha_slow=11;
        psd_slwalpha1(n,nn)=sum(sum (dd_stft1(n,nn,s_alpha_slow:e_alpha_slow,:))/(f(e_alpha_slow)-f(s_alpha_slow)));
        
        
        s_beta= 13;
        e_beta= 32;
        psd_beta1(n,nn)=sum(sum (dd_stft1(n,nn,s_beta:e_beta,:))/(f(e_beta)-f(s_beta)));
        
        
        s_gamma= 32;
        e_gamma= 64;
        psd_gamma1(n,nn)=sum(sum (dd_stft1(n,nn,s_gamma:e_gamma,:))/(f(e_gamma)-f(s_gamma)));
        
    end
end

%%
d=0;
e=[1:12 14 15 30:-1:19 18 17];
for i=1:40
    for b=1:14
                         
% power assymetry (subtracting)
psd_theta2(i,b)=sum(sum (dd_stft1(i,e(b),s_theta:e_theta,:)-dd_stft1(i,e(b+14),s_theta:e_theta,:))/(f(e_theta)-f(s_theta)));
psd_alpha2(i,b)=sum(sum (dd_stft1(i,e(b),s_alpha:e_alpha,:)-dd_stft1(i,e(b+14),s_alpha:e_alpha,:))/(f(e_alpha)-f(s_alpha)));
psd_beta2(i,b)=sum(sum (dd_stft1(i,e(b),s_beta:e_beta,:)-dd_stft1(i,e(b+14),s_beta:e_beta,:))/(f(e_beta)-f(s_beta)));
psd_gamma2(i,b)=sum(sum (dd_stft1(i,e(b),s_gamma:e_gamma,:)-dd_stft1(i,e(b+14),s_gamma:e_gamma,:))/(f(e_gamma)-f(s_gamma)));


% power assymetry (didiving)
psd_theta3(i,b)=sum(sum (dd_stft1(i,e(b),s_theta:e_theta,:)./dd_stft1(i,e(b+14),s_theta:e_theta,:))/(f(e_theta)-f(s_theta)));
psd_alpha3(i,b)=sum(sum (dd_stft1(i,e(b),s_alpha:e_alpha,:)./dd_stft1(i,e(b+14),s_alpha:e_alpha,:))/(f(e_alpha)-f(s_alpha)));
psd_beta3(i,b)=sum(sum (dd_stft1(i,e(b),s_beta:e_beta,:)./dd_stft1(i,e(b+14),s_beta:e_beta,:))/(f(e_beta)-f(s_beta)));
psd_gamma3(i,b)=sum(sum (dd_stft1(i,e(b),s_gamma:e_gamma,:)./dd_stft1(i,e(b+14),s_gamma:e_gamma,:))/(f(e_gamma)-f(s_gamma)));

    end 
end

temp_feature_table = cat(2,psd_alpha1,psd_alpha2,psd_alpha3,psd_slwalpha1,psd_beta1,psd_beta2,psd_beta3,psd_gamma1,psd_gamma2,psd_gamma3,psd_theta1,psd_theta2,psd_theta3);         

if m>1
     feature_table=cat(1,feature_table,temp_feature_table);
 else
     feature_table=temp_feature_table;
     end
clearvars DEAP temp_feature_table psd_alpha1 psd_alpha2 psd_alpha3 psd_slwalpha1 psd_beta1 psd_beta2 psd_beta3 psd_gamma1 psd_gamma2 psd_gamma3 psd_theta1 psd_theta2 psd_theta3 loadfile;    


end
