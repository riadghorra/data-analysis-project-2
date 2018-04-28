%data = csvread('DMI_data_full_option.csv');
clear all
%data= importdata('Skive_Billund_50_50.mat')
data= importdata('Billund_90_10.mat')
%%
%ICA
noc = 48
[H, W, Winv] = fastica (data,'numOfIC',noc,'approach','symm','g','tanh');
%%
%NMF
noc = 48
[W,H,L]=NMFPG(data,noc);
%%
%AA
noc = 48
[W,S,H,L]=ArchetypalAnalysis(data,noc);
%%
%Sparse Coding
lambda = .01;
noc = 48;
[W,H,L,L1]=SparseCoding(data,noc,lambda,[1 1]);

