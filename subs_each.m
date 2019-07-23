function [ s ] = subs_each( s, condition )
%Substitute condition to each member of structure
%   Detailed explanation goes here
    sf = fieldnames(s);
    for i=1:numel(sf)
        s=setfield(s, sf{i}, subs(getfield(s, sf{i}), condition));
    end
end

