%% ========================================================================
%  MÔ PHỎNG GIẢI PHÁP BẢO MẬT HẬU LƯỢNG TỬ CHO 5G/6G
%  A-QSF: Constrained Temporal CNN-DQN for Quantum-Safe 5G/6G Broadcast Services
%  Tác giả: Doãn Thanh Bình,
%  Mã nguồn tạo ra ket qua tai thu muc: AQSC_results_v5_matlab
% =========================================================================
%  Đánh giá Mean latency, P95 latency, security compliance, MEC crypto
%  utilization
% =========================================================================
clear;clc;close all;
cfg=AQSC_DefaultConfig_v5();if ~exist(cfg.outputDir,'dir'),mkdir(cfg.outputDir);end
profiles=AQSC_LoadHardwareBenchmark_v5(cfg.benchmarkCSV);writetable(struct2table(profiles),fullfile(cfg.outputDir,'action_profiles_v5.csv'));
[frozen,trainingLog]=AQSC_TrainAgents_v5(cfg,profiles);

% Manuscript Figure 4: show + export immediately after training.
AQSC_PlotFigure4_v5(trainingLog,cfg);

% Manuscript Figure 5: show + export directly from measured benchmark CSV.
AQSC_PlotFigure5_v5(cfg.benchmarkCSV,cfg);

results=AQSC_Evaluate_v5(frozen,cfg,profiles); %#ok<NASGU>
% AQSC_Evaluate_v5 now creates manuscript Figure 6 and leaves it visible.
fprintf('A-QSF v5 complete: %s\n',cfg.outputDir);
fprintf('Figure 4, Figure 5, and Figure 6 remain visible in MATLAB figure windows.\n');
