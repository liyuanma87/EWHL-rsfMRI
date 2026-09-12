clc;
clear;
close all;
tic
dbstop if error
script_dir = fileparts(mfilename('fullpath'));
data = importdata([Path filesep '884_feature143_centered_processed.xlsx']);
[subNum,featureNum] = size(data);
data=xlsread("884_feature143_centered_processed.xlsx");

a2=xlsread("884data.xlsx");

site = a2(:,2);
%TR = a2(:,5);
sex = a2(:,7);
age  = a2(:,6);
Ddata = combat(data(2:end,:).',sex,age,1);
%Ddata = combat(Ddata,TR,age,1);
Ddata = combat(Ddata,site,age,1);
Ddata = Ddata';
dlmwrite([Path filesep 'matlab_combat143.txt'], Ddata, 'delimiter' , ' ' , 'precision', '%0.4f');