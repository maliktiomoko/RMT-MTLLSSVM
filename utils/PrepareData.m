close all
clear
clc

addpath([pwd '/Functions'])

dataSetupParameters()

defineSplits();
retrieveSplitData();
pairing();