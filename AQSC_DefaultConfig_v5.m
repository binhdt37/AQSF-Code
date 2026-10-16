function cfg = AQSC_DefaultConfig_v5()
% Reproducible A-QSF v5 configuration: paired system-level evaluation,
% measured PQC timings, temporal CNN ablation, and hard action masking.
cfg.version = '5.0-Q2-four-upgrades';
cfg.masterSeed = 20260806;
cfg.outputDir = fullfile(pwd,'AQSC_results_v5_matlab');
cfg.benchmarkCSV = fullfile(fileparts(mfilename('fullpath')),'benchmark','pqc_hardware_benchmark.csv');

cfg.ueGrid = [500 1000 2000 4000 6000 8000 10000];
cfg.trainEpisodes = 420; cfg.trainFrames = 80;
cfg.evalFrames = 120; cfg.monteCarloRuns = 30;
cfg.controlInterval_s = 0.1; cfg.deadline_ms = 10;
cfg.bandwidthMHz = 100; cfg.carrierGHz = 3.5;
cfg.subcarrierSpacingKHz = 30; cfg.mimoLayers = 4;
cfg.packetBytes = 1500; cfg.offeredLoad_Mbps = 50;
cfg.mecCores = 32; cfg.targetCoverage = 0.95;
cfg.probeUEs = 384; cfg.cqiMargin_dB = 1.5;
cfg.channelModel = '3GPP TR 38.901 UMa large-scale model';
cfg.linkAdaptation = '3GPP TS 38.214 CQI spectral-efficiency table';
cfg.mbsArchitecture = '3GPP TS 23.247 system-level abstraction';

cfg.initialStateProb = [0.20 0.60 0.20];
cfg.transition = [0.965 0.025 0.010; 0.015 0.970 0.015; 0.010 0.035 0.955];
cfg.requiredCategory = [5 3 1];
cfg.threatBase = [0.92 0.48 0.20];

cfg.agent.History = 8; cfg.agent.HiddenDim = 32;
cfg.agent.ConvFilters = 10; cfg.agent.ConvKernel = 3;
cfg.agent.NumActions = 6; cfg.agent.LearningRate = 7e-4;
cfg.agent.DiscountFactor = 0.96; cfg.agent.Epsilon = 1.0;
cfg.agent.EpsilonMin = 0.05; cfg.agent.EpsilonDecay = 0.9985;
cfg.agent.BatchSize = 64; cfg.agent.ReplayCapacity = 30000;
cfg.agent.TargetUpdateInterval = 250; cfg.agent.GradientClip = 5;
cfg.violationPenalty = 12.0; cfg.bootstrapReplicates = 3000;
cfg.testSeeds = cfg.masterSeed + (2000:2000+cfg.monteCarloRuns-1);
end
