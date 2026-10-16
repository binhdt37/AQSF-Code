function mask = AQSC_ActionMask_v5(requiredCategory,profiles)
mask = reshape([profiles.Category] >= requiredCategory,1,[]);
assert(any(mask),'Safety mask produced an empty feasible set.');
end
