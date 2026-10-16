function met = AQSC_ProfileCost_v5(profile,nUE,capacity,bgCPU,threat,cfg)
r=profile.Refresh_s;
kemCore=(nUE/r)*profile.KEM_ms/1000;
sigCore=(profile.Sign_ms+profile.Verify_ms)/1000/r;
cryptoUtil=bgCPU+(kemCore+sigCore)/cfg.mecCores;
overhead=(nUE/r*profile.CiphertextBytes*8+profile.SignatureBytes*8/r)/1e6+ ...
 cfg.offeredLoad_Mbps*28/cfg.packetBytes;
usable=max(1,capacity-overhead);radioUtil=cfg.offeredLoad_Mbps/usable;
queue=.25*radioUtil/max(.03,1-min(radioUtil,.97));
cpuQueue=.12*cryptoUtil/max(.03,1-min(cryptoUtil,.97));
latency=.60+8*cfg.packetBytes/usable/1000+queue+cpuQueue;
if radioUtil>1,latency=latency+min(20,20*(radioUtil-1));end
loss=min(.95,max(0,1-usable/cfg.offeredLoad_Mbps)+max(0,cryptoUtil-.95)*.15);
met=struct('Latency_ms',latency,'Loss',loss,'Throughput_Mbps',min(usable,cfg.offeredLoad_Mbps)*(1-loss), ...
 'CryptoUtil',cryptoUtil,'Overhead_Mbps',overhead,'Exposure',threat*(r/120));
end
