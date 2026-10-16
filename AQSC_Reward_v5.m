function reward = AQSC_Reward_v5(met,switched,compliant,cfg)
reward=-(.25*min(met.Latency_ms/cfg.deadline_ms,2)+.16*met.Loss+ ...
 .14*min(met.CryptoUtil,1.5)+.08*min(met.Overhead_Mbps/20,2)+.35*met.Exposure+.02*switched);
if ~compliant,reward=reward-cfg.violationPenalty;end
end
