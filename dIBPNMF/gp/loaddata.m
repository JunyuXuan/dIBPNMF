function [ data ] = loaddata( )
%LOADDATA Summary of this function goes here
%   Detailed explanation goes here
    data =load('data.mat');
    data = data.data;
end

