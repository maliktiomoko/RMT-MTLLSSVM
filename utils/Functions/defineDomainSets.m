function domainSet = defineDomainSets(domainNames)

k = 0;
for s = 1 : length(domainNames)
    
    for t = 1 : length(domainNames)
        
        if t ~= s
            
            k = k + 1;
            domainSet{k} = {domainNames{s},domainNames{t}};
            
        end
        
    end
    
end

end

