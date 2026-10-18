#!/usr/bin/env python3
"""Independent executable mirror of the A-QSF v5 MATLAB experiment.

It is intentionally dependency-light (NumPy/Pandas/Matplotlib only) and writes
the exact tables/figures used to update the manuscript. The MATLAB package
contains the same channel, masking, action and reporting definitions.
"""
from __future__ import annotations

import json
import math
import zlib
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).with_name("mpl-cache")))
import matplotlib.pyplot as plt


@dataclass
class Config:
    master_seed: int = 20260806
    output_dir: str = "AQSC_results_v5"
    ue_grid: tuple = (500, 1000, 2000, 4000, 6000, 8000, 10000)
    train_episodes: int = 420
    train_frames: int = 80
    eval_frames: int = 120
    monte_carlo_runs: int = 30
    control_interval_s: float = 0.1
    deadline_ms: float = 10.0
    bandwidth_mhz: float = 100.0
    carrier_ghz: float = 3.5
    mimo_layers: int = 4
    packet_bytes: int = 1500
    offered_mbps: float = 50.0
    mec_cores: int = 32
    target_coverage: float = 0.95
    probe_ues: int = 384
    cqi_margin_db: float = 1.5
    bootstrap_replicates: int = 3000
    history: int = 8
    replay_capacity: int = 30000
    batch_size: int = 64
    learning_rate: float = 7e-4
    gamma: float = 0.96
    target_update: int = 250
    violation_penalty: float = 12.0


CFG = Config()
CQI_EFF = np.array([0.0, .1523, .2344, .3770, .6016, .8770, 1.1758, 1.4766,
                    1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152,
                    5.5547, 6.2266, 6.9141, 7.4063])
TRANSITION = np.array([[.965, .025, .010], [.015, .970, .015], [.010, .035, .955]])
INITIAL = np.array([.20, .60, .20])
REQUIRED = np.array([5, 3, 1])
THREAT_BASE = np.array([.92, .48, .20])


def load_benchmark(path: Path):
    b = pd.read_csv(path)
    med = {(r.algorithm, r.operation): float(r.median_ms) for r in b.itertuples()}
    sig_bytes = {r.algorithm: int(r.signature_bytes) for r in b.itertuples()
                 if r.operation == "sign" and not pd.isna(r.signature_bytes)}
    ct_bytes = {r.algorithm: int(r.ciphertext_bytes) for r in b.itertuples()
                if r.operation == "encapsulate" and not pd.isna(r.ciphertext_bytes)}
    return med, sig_bytes, ct_bytes


def build_profiles(bench_csv: Path):
    med, sig_bytes, ct_bytes = load_benchmark(bench_csv)
    # Two cryptoperiods per NIST category. Both members use the measured
    # fast SLH-DSA variant; the temporal policy trades exposure window against
    # KEM/signature service load without changing the required category.
    specs = [
        ("C5-short", 5, "ML-KEM-1024", "SLH-DSA-SHA2-256f", 10.0),
        ("C5-long",  5, "ML-KEM-1024", "SLH-DSA-SHA2-256f", 30.0),
        ("C3-short", 3, "ML-KEM-768",  "SLH-DSA-SHA2-192f", 20.0),
        ("C3-long",  3, "ML-KEM-768",  "SLH-DSA-SHA2-192f", 60.0),
        ("C1-short", 1, "ML-KEM-512",  "SLH-DSA-SHA2-128f", 30.0),
        ("C1-long",  1, "ML-KEM-512",  "SLH-DSA-SHA2-128f", 120.0),
    ]
    out = []
    for name, cat, kem, slh, refresh in specs:
        out.append(dict(name=name, category=cat, kem=kem, slh=slh, refresh_s=refresh,
                        kem_ms=med[(kem, "encapsulate")]+med[(kem, "decapsulate")],
                        sign_ms=med[(slh, "sign")], verify_ms=med[(slh, "verify")],
                        ct_bytes=ct_bytes[kem], sig_bytes=sig_bytes[slh]))
    return out


def uma_pathloss(d2d_m, los, fc_ghz=3.5, hut=1.5, hbs=25.0):
    d3d = np.sqrt(d2d_m**2 + (hbs-hut)**2)
    pl_los = 28.0 + 22.0*np.log10(d3d) + 20.0*np.log10(fc_ghz)
    pl_nlos = 13.54 + 39.08*np.log10(d3d) + 20.0*np.log10(fc_ghz) - .6*(hut-1.5)
    return np.where(los, pl_los, np.maximum(pl_los, pl_nlos))


def los_probability(d2d_m):
    d = np.maximum(d2d_m, 1.0)
    return np.minimum(18.0/d, 1.0)*(1.0-np.exp(-d/63.0)) + np.exp(-d/63.0)


def cqi_capacity(sinr_db, cfg=CFG):
    thresholds = 10*np.log10(np.maximum(2**(CQI_EFF/.72)-1, 1e-12)) + cfg.cqi_margin_db
    idx = np.searchsorted(thresholds, sinr_db, side="right") - 1
    idx = int(np.clip(idx, 0, len(CQI_EFF)-1))
    return cfg.bandwidth_mhz*cfg.mimo_layers*CQI_EFF[idx]*.82, idx


def generate_trace(seed: int, n_ue: int, frames: int, cfg=CFG):
    rng = np.random.default_rng(seed + 17*n_ue)
    state = np.zeros(frames, dtype=int)
    state[0] = rng.choice(3, p=INITIAL)
    for k in range(1, frames): state[k] = rng.choice(3, p=TRANSITION[state[k-1]])
    threat = np.empty(frames); threat[0] = THREAT_BASE[state[0]] + .07*rng.normal()
    for k in range(1, frames):
        threat[k] = .84*threat[k-1] + .16*THREAT_BASE[state[k]] + .045*rng.normal()
    threat = np.clip(threat, 0, 1)
    anomaly = np.clip(.08 + .78*(state==0) + .10*rng.normal(size=frames), 0, 1)

    m = min(n_ue, cfg.probe_ues)
    radius = 35 + (500-35)*np.sqrt(rng.random(m))
    angle = 2*np.pi*rng.random(m)
    x, y = radius*np.cos(angle), radius*np.sin(angle)
    los = rng.random(m) < los_probability(radius)
    shadow = rng.normal(0, np.where(los, 4.0, 6.0), m)
    neighbor = np.array([[1000*np.cos(a), 1000*np.sin(a)] for a in np.arange(6)*np.pi/3])
    caps, cqis, p5sinr = np.empty(frames), np.empty(frames, int), np.empty(frames)
    bg = np.empty(frames); bg[0] = np.clip(.20 + .15*n_ue/10000 + .08*rng.normal(), .05, .75)
    for k in range(frames):
        if k:
            angle += rng.normal(0, .0015, m)
            x, y = radius*np.cos(angle), radius*np.sin(angle)
            shadow = .96*shadow + np.sqrt(1-.96**2)*rng.normal(0, np.where(los,4.0,6.0),m)
            bg[k] = np.clip(.92*bg[k-1] + .08*(.20+.25*n_ue/10000) + .035*rng.normal(), .05, .90)
        desired_dbm = 46.0 + 8.0 - uma_pathloss(radius, los, cfg.carrier_ghz) - shadow
        d_int = np.sqrt((x[:,None]-neighbor[:,0])**2 + (y[:,None]-neighbor[:,1])**2)
        int_pl = uma_pathloss(np.maximum(d_int,35), np.zeros_like(d_int,dtype=bool), cfg.carrier_ghz)
        interf_mw = np.sum(10**((46+8-int_pl-7)/10), axis=1)  # 20% activity ~= -7 dB
        noise_mw = 10**((-174 + 10*np.log10(cfg.bandwidth_mhz*1e6) + 7)/10)
        sinr = 10*np.log10(10**(desired_dbm/10)/(interf_mw+noise_mw))
        p5sinr[k] = np.quantile(sinr, 1-cfg.target_coverage)
        caps[k], cqis[k] = cqi_capacity(p5sinr[k], cfg)
    return dict(state=state, required=REQUIRED[state], threat=threat, anomaly=anomaly,
                capacity=caps, cqi=cqis, p5sinr=p5sinr, background_cpu=bg)


def valid_mask(required, profiles):
    return np.array([p["category"] >= required for p in profiles], dtype=bool)


def profile_cost(profile, n_ue, cap_mbps, bg_cpu, threat, cfg=CFG):
    r = profile["refresh_s"]
    kem_core = (n_ue/r)*profile["kem_ms"]/1000
    sig_core = (profile["sign_ms"]+profile["verify_ms"])/1000/r
    crypto_util = bg_cpu + (kem_core+sig_core)/cfg.mec_cores
    overhead = (n_ue/r*profile["ct_bytes"]*8 + profile["sig_bytes"]*8/r)/1e6 + cfg.offered_mbps*28/cfg.packet_bytes
    usable = max(1.0, cap_mbps-overhead)
    radio_util = cfg.offered_mbps/usable
    queue = .25*radio_util/max(.03,1-min(radio_util,.97))
    cpu_queue = .12*crypto_util/max(.03,1-min(crypto_util,.97))
    latency = .60 + 8*cfg.packet_bytes/usable/1000 + queue + cpu_queue
    if radio_util > 1: latency += min(20.0, 20*(radio_util-1))
    loss = min(.95, max(0.0, 1-usable/cfg.offered_mbps) + max(0,crypto_util-.95)*.15)
    delivered = min(usable, cfg.offered_mbps)*(1-loss)
    # Threat-weighted exposure-window term makes short/long cryptoperiods a
    # real decision while category masking remains the hard safety constraint.
    exposure = threat*(r/120.0)
    return dict(latency_ms=latency, loss=loss, throughput=delivered,
                crypto_util=crypto_util, overhead_mbps=overhead, exposure=exposure)


def make_observation(prev, trace, k, cfg=CFG):
    cap = trace["capacity"][k]
    return np.array([
        min(prev["latency_ms"]/cfg.deadline_ms, 2)/2,
        min(prev["loss"]/.10, 1),
        min(prev["throughput"]/max(cap,1), 1),
        np.clip((trace["p5sinr"][k]+10)/30,0,1),
        np.clip(trace["threat"][k] + .10*np.sin(k*.7),0,1),
        trace["anomaly"][k],
        min(prev["crypto_util"],1),
        np.clip(trace["background_cpu"][k],0,1),
    ], dtype=float)


class Adam:
    def __init__(self, params, lr):
        self.lr=lr; self.t=0
        self.m={k:np.zeros_like(v) for k,v in params.items()}
        self.v={k:np.zeros_like(v) for k,v in params.items()}
    def step(self, params, grads):
        self.t += 1
        for k in params:
            g=np.clip(grads[k],-5,5); self.m[k]=.9*self.m[k]+.1*g; self.v[k]=.999*self.v[k]+.001*g*g
            mh=self.m[k]/(1-.9**self.t); vh=self.v[k]/(1-.999**self.t)
            params[k] -= self.lr*mh/(np.sqrt(vh)+1e-8)


class DenseNet:
    def __init__(self, rng, input_dim=8, hidden=32, actions=6):
        self.p={"W1":rng.normal(0,np.sqrt(2/input_dim),(input_dim,hidden)),"b1":np.zeros(hidden),
                "W2":rng.normal(0,np.sqrt(2/hidden),(hidden,actions)),"b2":np.zeros(actions)}
    def clone(self):
        z=object.__new__(DenseNet); z.p={k:v.copy() for k,v in self.p.items()}; return z
    def forward(self,x,cache=False):
        z=x@self.p["W1"]+self.p["b1"]; h=np.maximum(z,0); q=h@self.p["W2"]+self.p["b2"]
        return (q,(x,z,h)) if cache else q
    def backward(self,c,dq):
        x,z,h=c; dz=(dq@self.p["W2"].T)*(z>0)
        return {"W2":h.T@dq,"b2":dq.sum(0),"W1":x.T@dz,"b1":dz.sum(0)}


class TemporalCNNNet:
    def __init__(self,rng,history=8,channels=8,filters=10,kernel=3,hidden=32,actions=6):
        self.history=history; self.channels=channels; self.kernel=kernel
        self.p={"K":rng.normal(0,np.sqrt(2/(kernel*channels)),(filters,kernel,channels)),"bc":np.zeros(filters),
                "W1":rng.normal(0,np.sqrt(2/(filters+channels)),(filters+channels,hidden)),"b1":np.zeros(hidden),
                "W2":rng.normal(0,np.sqrt(2/hidden),(hidden,actions)),"b2":np.zeros(actions)}
    def clone(self):
        z=object.__new__(TemporalCNNNet); z.history=self.history;z.channels=self.channels;z.kernel=self.kernel
        z.p={k:v.copy() for k,v in self.p.items()}; return z
    def forward(self,x,cache=False):
        n=x.shape[0]; L=self.history-self.kernel+1; conv=np.empty((n,L,self.p["K"].shape[0]))
        for t in range(L): conv[:,t,:]=np.einsum('ntc,ftc->nf',x[:,t:t+self.kernel,:],self.p["K"])+self.p["bc"]
        relu=np.maximum(conv,0); pooled=relu.mean(1); feat=np.concatenate([pooled,x[:,-1,:]],1)
        z=feat@self.p["W1"]+self.p["b1"]; h=np.maximum(z,0); q=h@self.p["W2"]+self.p["b2"]
        return (q,(x,conv,relu,feat,z,h)) if cache else q
    def backward(self,c,dq):
        x,conv,relu,feat,z,h=c; n=x.shape[0]; L=conv.shape[1]
        g={}; g["W2"]=h.T@dq;g["b2"]=dq.sum(0); dz=(dq@self.p["W2"].T)*(z>0)
        g["W1"]=feat.T@dz;g["b1"]=dz.sum(0); df=dz@self.p["W1"].T
        dconv=(df[:,:self.p["K"].shape[0]][:,None,:]/L)*(conv>0)
        g["K"]=np.zeros_like(self.p["K"]); g["bc"]=dconv.sum((0,1))
        for t in range(L): g["K"] += np.einsum('nf,ntc->ftc',dconv[:,t,:],x[:,t:t+self.kernel,:])
        return g


class DQNAgent:
    def __init__(self, kind, seed, masked=True, cfg=CFG):
        self.kind=kind;self.masked=masked;self.cfg=cfg;self.rng=np.random.default_rng(seed)
        self.online=TemporalCNNNet(self.rng,history=cfg.history) if kind=="cnn" else DenseNet(self.rng)
        self.target=self.online.clone();self.opt=Adam(self.online.p,cfg.learning_rate);self.learn_steps=0
        self.buf=[];self.pos=0;self.epsilon=1.0
    def features(self,hist): return hist[None,:,:] if self.kind=="cnn" else hist[-1:,:]
    def select(self,hist,mask,training=False,return_proposal=False):
        q=self.online.forward(self.features(hist))[0]; proposal=int(np.argmax(q)); legal=np.flatnonzero(mask)
        if training and self.rng.random()<self.epsilon: proposal=int(self.rng.integers(len(q)))
        action=proposal
        if self.masked and not mask[action]:
            action=int(legal[np.argmax(q[legal])]) if not (training and self.rng.random()<self.epsilon) else int(self.rng.choice(legal))
        elif self.masked and training and self.rng.random()<self.epsilon: action=int(self.rng.choice(legal))
        if training: self.epsilon=max(.05,self.epsilon*.9985)
        return (action,proposal) if return_proposal else action
    def remember(self,h,a,r,hn,done,maskn):
        item=(h.copy(),a,r,hn.copy(),done,maskn.copy())
        if len(self.buf)<self.cfg.replay_capacity:self.buf.append(item)
        else:self.buf[self.pos]=item
        self.pos=(self.pos+1)%self.cfg.replay_capacity
    def learn(self):
        if len(self.buf)<self.cfg.batch_size:return
        idx=self.rng.choice(len(self.buf),self.cfg.batch_size,replace=False); batch=[self.buf[i] for i in idx]
        h=np.stack([x[0] for x in batch]); hn=np.stack([x[3] for x in batch])
        x=h if self.kind=="cnn" else h[:,-1,:]; xn=hn if self.kind=="cnn" else hn[:,-1,:]
        a=np.array([x[1] for x in batch]); r=np.array([x[2] for x in batch]); done=np.array([x[4] for x in batch])
        mn=np.stack([x[5] for x in batch]); qn=self.online.forward(xn); qn=np.where(mn,qn,-1e30) if self.masked else qn
        an=np.argmax(qn,1); qt=self.target.forward(xn); y=r+self.cfg.gamma*(1-done)*qt[np.arange(len(batch)),an]
        q,c=self.online.forward(x,True); err=q[np.arange(len(batch)),a]-y; dq=np.zeros_like(q);dq[np.arange(len(batch)),a]=2*err/len(batch)
        self.opt.step(self.online.p,self.online.backward(c,dq));self.learn_steps+=1
        if self.learn_steps%self.cfg.target_update==0:self.target=self.online.clone()


def reward_for(met, profile, threat, switched, compliant, cfg=CFG):
    risk=met["exposure"]
    utility=-(.25*min(met["latency_ms"]/cfg.deadline_ms,2)+.16*met["loss"]+
              .14*min(met["crypto_util"],1.5)+.08*min(met["overhead_mbps"]/20,2)+
              .35*risk+.02*switched)
    if not compliant: utility -= cfg.violation_penalty
    return utility


def train_agent(kind,profiles,masked,seed,cfg=CFG):
    agent=DQNAgent(kind,seed,masked,cfg); returns=[]; compliance=[]
    for ep in range(cfg.train_episodes):
        n_ue=int(agent.rng.choice(cfg.ue_grid)); trace=generate_trace(seed+10000+ep,n_ue,cfg.train_frames,cfg)
        prev=dict(latency_ms=2.0,loss=.001,throughput=cfg.offered_mbps,crypto_util=.25)
        hist=np.zeros((cfg.history,8)); last=3; ret=0; ok=0
        for k in range(cfg.train_frames):
            obs=make_observation(prev,trace,k,cfg); hist=np.vstack([hist[1:],obs]); mask=valid_mask(trace["required"][k],profiles)
            action=agent.select(hist,mask,True); met=profile_cost(profiles[action],n_ue,trace["capacity"][k],trace["background_cpu"][k],trace["threat"][k],cfg)
            compliant=profiles[action]["category"]>=trace["required"][k]; rew=reward_for(met,profiles[action],trace["threat"][k],action!=last,compliant,cfg)
            prev=met; next_obs=make_observation(prev,trace,min(k+1,cfg.train_frames-1),cfg); hn=np.vstack([hist[1:],next_obs])
            nextmask=valid_mask(trace["required"][min(k+1,cfg.train_frames-1)],profiles)
            agent.remember(hist,action,rew,hn,k==cfg.train_frames-1,nextmask);agent.learn();ret+=rew;ok+=compliant;last=action
        returns.append(ret);compliance.append(100*ok/cfg.train_frames)
        if (ep+1)%30==0: print(f'  {kind}/{"mask" if masked else "nomask"} episode {ep+1}/{cfg.train_episodes}',flush=True)
    return agent,pd.DataFrame({"Episode":np.arange(1,cfg.train_episodes+1),"Return":returns,"Compliance_pct":compliance,"Agent":f'{kind}-masked' if masked else f'{kind}-unmasked'})


def heuristic_action(trace,k,mask,profiles):
    legal=np.flatnonzero(mask); threat=trace["threat"][k]; cpu=trace["background_cpu"][k]; cap=trace["capacity"][k]
    # Short cryptoperiod during elevated risk; long during pressure.
    preferred_short = threat>.70 and cpu<.78 and cap>CFG.offered_mbps*1.05
    for a in legal:
        if ("short" in profiles[a]["name"])==preferred_short:return int(a)
    return int(legal[-1])


def greedy_action(trace,k,mask,profiles,n_ue,last,cfg=CFG):
    """Model-based one-step reference: best immediate utility among feasible actions.
    Uses the exact cost model and the current threat/capacity/background load, i.e.
    information the learned controller does not have. Not a deployable comparator."""
    best=None
    for a in np.flatnonzero(mask):
        met=profile_cost(profiles[a],n_ue,trace["capacity"][k],trace["background_cpu"][k],trace["threat"][k],cfg)
        r=reward_for(met,profiles[a],trace["threat"][k],a!=last,True,cfg)
        if best is None or r>best[0]: best=(r,int(a))
    return best[1]


def evaluate(agents,profiles,cfg=CFG):
    methods=["CNN-DQN+Mask","DQN-only+Mask","CNN-DQN penalty-only","Heuristic+Mask","Random-feasible","Static-C5-long","Static-C5-short","One-step greedy"]
    rows=[]; frames=[]
    for ir in range(cfg.monte_carlo_runs):
      seed=cfg.master_seed+2000+ir
      for n_ue in cfg.ue_grid:
        trace=generate_trace(seed,n_ue,cfg.eval_frames,cfg)
        for method in methods:
          prev=dict(latency_ms=2.0,loss=.001,throughput=cfg.offered_mbps,crypto_util=.25);hist=np.zeros((cfg.history,8));last=3
          vals=[]; proposals=0; interventions=0
          rng=np.random.default_rng(seed+zlib.crc32(method.encode())%10000+n_ue)  # stable across runs (hash() is salted)
          for k in range(cfg.eval_frames):
            obs=make_observation(prev,trace,k,cfg);hist=np.vstack([hist[1:],obs]);mask=valid_mask(trace["required"][k],profiles)
            if method=="CNN-DQN+Mask": action,proposal=agents["cnn_mask"].select(hist,mask,False,True)
            elif method=="DQN-only+Mask": action,proposal=agents["dense_mask"].select(hist,mask,False,True)
            elif method=="CNN-DQN penalty-only": action,proposal=agents["cnn_unmasked"].select(hist,np.ones(6,dtype=bool),False,True)
            elif method=="Heuristic+Mask": action=proposal=heuristic_action(trace,k,mask,profiles)
            elif method=="Random-feasible": action=proposal=int(rng.choice(np.flatnonzero(mask)))
            elif method=="Static-C5-long": action=proposal=1
            elif method=="Static-C5-short": action=proposal=0
            else: action=proposal=greedy_action(trace,k,mask,profiles,n_ue,last,cfg)
            proposals += int(not mask[proposal]); interventions += int(action!=proposal)
            met=profile_cost(profiles[action],n_ue,trace["capacity"][k],trace["background_cpu"][k],trace["threat"][k],cfg)
            compliant=profiles[action]["category"]>=trace["required"][k];switched=action!=last
            met.update(compliant=compliant,action=action,switched=switched,required=trace["required"][k])
            met["utility"]=reward_for(met,profiles[action],trace["threat"][k],switched,compliant,cfg)
            vals.append(met);frames.append(dict(Run=ir+1,Seed=seed,UE=n_ue,Method=method,Frame=k+1,
                RequiredCategory=trace["required"][k],Action=action+1,ActionName=profiles[action]["name"],
                Latency_ms=met["latency_ms"],Throughput_Mbps=met["throughput"],Loss_pct=100*met["loss"],
                Compliance=int(compliant),CryptoUtil_pct=100*met["crypto_util"],CQI=int(trace["cqi"][k]),
                CellEdgeSINR_dB=trace["p5sinr"][k],Capacity_Mbps=trace["capacity"][k]))
            prev=met;last=action
          lat=np.array([x["latency_ms"] for x in vals]);loss=np.array([x["loss"] for x in vals]);thr=np.array([x["throughput"] for x in vals])
          rows.append(dict(Run=ir+1,Seed=seed,UE=n_ue,Method=method,MeanLatency_ms=lat.mean(),P95Latency_ms=np.quantile(lat,.95),
              P99Latency_ms=np.quantile(lat,.99),MeanThroughput_Mbps=thr.mean(),MeanLoss_pct=100*loss.mean(),
              DeadlineMiss_pct=100*np.mean(lat>cfg.deadline_ms),SecurityCompliance_pct=100*np.mean([x["compliant"] for x in vals]),
              MeanCryptoUtil_pct=100*np.mean([x["crypto_util"] for x in vals]),MeanExposure=np.mean([x["exposure"] for x in vals]),
              MeanCryptoperiod_s=np.mean([profiles[x["action"]]["refresh_s"] for x in vals]),SwitchCount=np.sum([x["switched"] for x in vals]),
              InvalidProposal_pct=100*proposals/cfg.eval_frames,MaskIntervention_pct=100*interventions/cfg.eval_frames,MeanUtility=np.mean([x["utility"] for x in vals])))
    return pd.DataFrame(rows),pd.DataFrame(frames)


def paired_statistics(summary,cfg=CFG):
    comparators=["DQN-only+Mask","Heuristic+Mask","Static-C5-long","Static-C5-short","One-step greedy","CNN-DQN penalty-only"]
    metrics=[("MeanLatency_ms",-1),("P95Latency_ms",-1),("MeanThroughput_Mbps",1),("MeanLoss_pct",-1),
             ("DeadlineMiss_pct",-1),("SecurityCompliance_pct",1),("MeanCryptoUtil_pct",-1),("MeanExposure",-1),("MeanUtility",1)]
    out=[]; rng=np.random.default_rng(cfg.master_seed+99)
    for ue in cfg.ue_grid:
      a=summary[(summary.UE==ue)&(summary.Method=="CNN-DQN+Mask")].sort_values("Seed")
      for comp in comparators:
       b=summary[(summary.UE==ue)&(summary.Method==comp)].sort_values("Seed")
       for metric,sign in metrics:
        d=sign*(a[metric].to_numpy()-b[metric].to_numpy());boots=np.mean(rng.choice(d,(cfg.bootstrap_replicates,len(d)),replace=True),1)
        p=stats.ttest_rel(sign*a[metric],sign*b[metric]).pvalue
        try: wp=stats.wilcoxon(d,alternative="greater").pvalue
        except ValueError: wp=1.0
        out.append(dict(UE=ue,Outcome=metric,Comparator=comp,CNNMean=a[metric].mean(),ComparatorMean=b[metric].mean(),
            Benefit= d.mean(),CI95Low=np.quantile(boots,.025),CI95High=np.quantile(boots,.975),
            EffectDz=d.mean()/max(d.std(ddof=1),1e-12),PairedT_P=p,WilcoxonOneSided_P=wp))
    stat=pd.DataFrame(out);stat["HolmP"]=1.0
    for comp in comparators:
      idx=stat.Comparator==comp; p=stat.loc[idx,"WilcoxonOneSided_P"].to_numpy();order=np.argsort(p);adj=np.empty_like(p);run=0
      for rank,j in enumerate(order): run=max(run,(len(p)-rank)*p[j]);adj[j]=min(1,run)
      stat.loc[idx,"HolmP"]=adj
    return stat


def aggregate_paired_statistics(summary,cfg=CFG):
    """Seed-level inference: average each outcome across the loads within a seed, then
    paired differences (positive Benefit = CNN-DQN+Mask better), bootstrap CI, d_z,
    one-sided Wilcoxon, Holm correction within comparator."""
    comparators=["DQN-only+Mask","Heuristic+Mask","Static-C5-long","Static-C5-short","One-step greedy","CNN-DQN penalty-only"]
    metrics=[("MeanLatency_ms",-1),("P95Latency_ms",-1),("MeanCryptoUtil_pct",-1),("MeanExposure",-1),("MeanUtility",1)]
    seedmean=summary.groupby(["Seed","Method"],as_index=False)[[m for m,_ in metrics]].mean()
    out=[]; rng=np.random.default_rng(cfg.master_seed+199)
    for comp in comparators:
      a=seedmean[seedmean.Method=="CNN-DQN+Mask"].sort_values("Seed"); b=seedmean[seedmean.Method==comp].sort_values("Seed")
      for metric,sign in metrics:
        d=sign*(a[metric].to_numpy()-b[metric].to_numpy()); boots=np.mean(rng.choice(d,(cfg.bootstrap_replicates,len(d)),replace=True),1)
        try: wp=stats.wilcoxon(d,alternative="greater").pvalue
        except ValueError: wp=1.0
        out.append(dict(Outcome=metric,Comparator=comp,CNNMean=a[metric].mean(),ComparatorMean=b[metric].mean(),Benefit=d.mean(),
            CI95Low=np.quantile(boots,.025),CI95High=np.quantile(boots,.975),EffectDz=d.mean()/max(d.std(ddof=1),1e-12),
            PairedT_P=stats.ttest_rel(sign*a[metric],sign*b[metric]).pvalue,WilcoxonOneSided_P=wp))
    st=pd.DataFrame(out); st["HolmP"]=1.0
    for comp in comparators:
      idx=st.Comparator==comp; p=st.loc[idx,"WilcoxonOneSided_P"].to_numpy(); order=np.argsort(p); adj=np.empty_like(p); run=0
      for rank,j in enumerate(order): run=max(run,(len(p)-rank)*p[j]); adj[j]=min(1,run)
      st.loc[idx,"HolmP"]=adj
    return st


def per_action_cost_table(profiles,cfg=CFG,loads=(500,10000)):
    """Cost of every action on identical evaluation traces (Table VII)."""
    rows=[]
    for n in loads:
        acc={p["name"]:dict(ov=[],cu=[],lat=[],exp=[]) for p in profiles}
        for ir in range(cfg.monte_carlo_runs):
            tr=generate_trace(cfg.master_seed+2000+ir,n,cfg.eval_frames,cfg)
            for k in range(cfg.eval_frames):
                for p in profiles:
                    c=profile_cost(p,n,tr["capacity"][k],tr["background_cpu"][k],tr["threat"][k],cfg); a=acc[p["name"]]
                    a["ov"].append(c["overhead_mbps"]);a["cu"].append(100*c["crypto_util"]);a["lat"].append(c["latency_ms"]);a["exp"].append(c["exposure"])
        for p in profiles:
            a=acc[p["name"]]; r=p["refresh_s"]
            rows.append(dict(Receivers=n,Action=p["name"],Category=p["category"],Cryptoperiod_s=r,KeyDist_Mbps=np.mean(a["ov"]),
                KEM_CPU_pct=100*((n/r)*p["kem_ms"]/1000)/cfg.mec_cores,Sign_CPU_pct=100*((p["sign_ms"]+p["verify_ms"])/1000/r)/cfg.mec_cores,
                MEC_util_pct=np.mean(a["cu"]),Latency_ms=np.mean(a["lat"]),ExposurePerThreat=r/120,Exposure=np.mean(a["exp"])))
    return pd.DataFrame(rows)


def make_figures(summary,training,outdir:Path,cfg=CFG):
    order=["CNN-DQN+Mask","DQN-only+Mask","Heuristic+Mask","Static-C5-long","Static-C5-short","One-step greedy"]
    colors=["#0B6E4F","#1F77B4","#E69F00","#7F7F7F","#C44E52","#8E5EA2"]
    fig,axs=plt.subplots(2,2,figsize=(10.4,5.8),constrained_layout=True)
    for ax,metric,label in zip(axs.flat,["MeanLatency_ms","P95Latency_ms","SecurityCompliance_pct","MeanCryptoUtil_pct"],
                               ["Mean latency (ms)","P95 latency (ms)","Security compliance (%)","MEC crypto utilization (%)"]):
      for method,c in zip(order,colors):
        g=summary[summary.Method==method].groupby("UE")[metric]
        mu=g.mean();ci=1.96*g.std()/np.sqrt(g.count());ax.errorbar(mu.index,mu,yerr=ci,marker='o',lw=1.5,label=method,color=c)
      ax.set_xlabel("Broadcast receivers");ax.set_ylabel(label);ax.grid(alpha=.25)
    axs[0,0].legend(fontsize=7);fig.savefig(outdir/"Figure_system_level_v5.png",dpi=300);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9.4,4.2),constrained_layout=True)
    for name,c in [("cnn-masked","#0B6E4F"),("dense-masked","#1F77B4")]:
      x=training[training.Agent==name]; y=x.Return.rolling(20,min_periods=1).mean();axs[0].plot(x.Episode,y,label=name,color=c)
    axs[0].set(xlabel="Training episode",ylabel="20-episode mean return",ylim=(-16,-5));axs[0].grid(alpha=.25);axs[0].legend(fontsize=8)
    for name,c in [("cnn-masked","#0B6E4F"),("dense-masked","#1F77B4"),("cnn-unmasked","#C44E52")]:
      x=training[training.Agent==name]; y=x.Compliance_pct.rolling(20,min_periods=1).mean();axs[1].plot(x.Episode,y,label=name,color=c)
    axs[1].set(xlabel="Training episode",ylabel="20-episode mean compliance (%)",ylim=(70,101));axs[1].grid(alpha=.25);axs[1].legend(fontsize=8)
    fig.savefig(outdir/"Figure_training_ablation_v5.png",dpi=300);plt.close(fig)


def main():
    package_root=Path(__file__).resolve().parent.parent
    out=package_root/"AQSC_results_v6_reproduced";out.mkdir(exist_ok=True)
    bench=package_root/"benchmark"/"pqc_hardware_benchmark.csv";profiles=build_profiles(bench)
    agents={};logs=[]
    for key,kind,masked,offset in [("cnn_mask","cnn",True,1),("dense_mask","dense",True,2),("cnn_unmasked","cnn",False,3)]:
        print("training",key,flush=True);agents[key],log=train_agent(kind,profiles,masked,CFG.master_seed+offset,CFG);logs.append(log)
    training=pd.concat(logs,ignore_index=True);summary,frames=evaluate(agents,profiles,CFG);stat=paired_statistics(summary,CFG)
    ablation=summary.groupby("Method",as_index=False).agg(MeanLatency_ms=("MeanLatency_ms","mean"),P95Latency_ms=("P95Latency_ms","mean"),
        Throughput_Mbps=("MeanThroughput_Mbps","mean"),Compliance_pct=("SecurityCompliance_pct","mean"),CryptoUtil_pct=("MeanCryptoUtil_pct","mean"),
        DeadlineMiss_pct=("DeadlineMiss_pct","mean"),MeanExposure=("MeanExposure","mean"),MeanCryptoperiod_s=("MeanCryptoperiod_s","mean"),
        InvalidProposal_pct=("InvalidProposal_pct","mean"),MaskIntervention_pct=("MaskIntervention_pct","mean"),MeanUtility=("MeanUtility","mean"),SwitchCount=("SwitchCount","mean"))
    pd.DataFrame(profiles).to_csv(out/"action_profiles_v5.csv",index=False);training.to_csv(out/"training_log_v5.csv",index=False)
    summary.to_csv(out/"run_level_summary_v5.csv",index=False);frames.to_csv(out/"frame_level_trace_v5.csv",index=False)
    stat.to_csv(out/"paired_statistics_v5.csv",index=False);ablation.to_csv(out/"ablation_summary_v5.csv",index=False)
    agg=aggregate_paired_statistics(summary,CFG);agg.to_csv(out/"aggregate_paired_statistics_v6.csv",index=False)
    order=["CNN-DQN+Mask","DQN-only+Mask","CNN-DQN penalty-only","Heuristic+Mask","Random-feasible","Static-C5-long","Static-C5-short","One-step greedy"]
    t5=summary.groupby("Method")[["SecurityCompliance_pct","MeanExposure","MeanCryptoUtil_pct","MeanUtility"]].mean().loc[order].reset_index()
    t5.columns=["Policy","Compliance_pct","Exposure","MEC_util_pct","Utility"];t5.to_csv(out/"table_V_policy_ablation_v6.csv",index=False)
    t6=summary[summary.UE==10000].groupby("Method")[["MeanLatency_ms","P95Latency_ms","DeadlineMiss_pct","SecurityCompliance_pct","MeanUtility"]].mean().loc[order].reset_index()
    t6.columns=["Policy","MeanLatency_ms","P95Latency_ms","DeadlineMiss_pct","Compliance_pct","Utility"];t6.to_csv(out/"table_VI_10000_receivers_v6.csv",index=False)
    per_action_cost_table(profiles,CFG).to_csv(out/"table_VII_per_action_cost_v6.csv",index=False)
    with open(out/"configuration_v5.json","w") as f: json.dump(asdict(CFG),f,indent=2)
    make_figures(summary,training,out,CFG);print(ablation.to_string(index=False));


if __name__=="__main__": main()
