function profiles = AQSC_LoadHardwareBenchmark_v5(csvPath)
% Import medians measured by pqc_hardware_benchmark.mjs. The benchmark is
% single-thread, warm-cache, pure JavaScript and is not presented as an
% optimized native-library result.
assert(isfile(csvPath),'Benchmark CSV not found: %s',csvPath);
T = readtable(csvPath,'TextType','string');
spec = {
 'C5-short',5,'ML-KEM-1024','SLH-DSA-SHA2-256f',10;
 'C5-long', 5,'ML-KEM-1024','SLH-DSA-SHA2-256f',30;
 'C3-short',3,'ML-KEM-768', 'SLH-DSA-SHA2-192f',20;
 'C3-long', 3,'ML-KEM-768', 'SLH-DSA-SHA2-192f',60;
 'C1-short',1,'ML-KEM-512', 'SLH-DSA-SHA2-128f',30;
 'C1-long', 1,'ML-KEM-512', 'SLH-DSA-SHA2-128f',120};
profiles = repmat(struct('Name','','Category',0,'KEM','','SLHDSA','', ...
 'Refresh_s',0,'KEM_ms',0,'Sign_ms',0,'Verify_ms',0,'CiphertextBytes',0,'SignatureBytes',0),6,1);
for i=1:size(spec,1)
 p=profiles(i); p.Name=spec{i,1}; p.Category=spec{i,2}; p.KEM=spec{i,3};
 p.SLHDSA=spec{i,4}; p.Refresh_s=spec{i,5};
 p.KEM_ms = value(T,p.KEM,'encapsulate','median_ms') + value(T,p.KEM,'decapsulate','median_ms');
 p.Sign_ms = value(T,p.SLHDSA,'sign','median_ms'); p.Verify_ms=value(T,p.SLHDSA,'verify','median_ms');
 p.CiphertextBytes=value(T,p.KEM,'encapsulate','ciphertext_bytes');
 p.SignatureBytes=value(T,p.SLHDSA,'sign','signature_bytes'); profiles(i)=p;
end
end
function x=value(T,alg,op,var)
 idx=T.algorithm==string(alg) & T.operation==string(op); assert(nnz(idx)==1,'Missing benchmark row: %s/%s',alg,op);
 x=T.(var)(idx); if iscell(x) || isstring(x), x=str2double(x); end; x=double(x);
end
