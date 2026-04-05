% % 数据拟合

tic;

for filenum = 7


    %文件地址：./10-30_single_sig/0001.mat   ./01-03double_matlab/00%02d.mat
    

    filePath = sprintf("./04-04overlap_matlab/00%02d.mat",filenum);
    signal1 = Signal.fromFile(filePath);
    signal1.data = -signal1.data;
    signal1 = f_remove_dc(signal1);
    % signal1.plot();
    signal1_energy = f_moving_energy(signal1, 1e-6, 'Plot', false);
    sig_frags = f_detect_pulses(signal1, "RefSig", signal1_energy, ...
        "Method", "mean_std", "Threshold", 5, "Plot", false);
    
    % 初始化特征时刻数组（原信号时间轴下，单位 s）
    % 仅存放对应类型的片段；拟合失败时存 NaN 占位
    t_sync = [];   % 同步光特征时刻
    t_scan = [];   % 扫描光特征时刻

    
    N = length(sig_frags);
     
    % 预分配：每个迭代一个位置，默认 NaN
    t_scan_all = NaN(1, N);
    t_sync_all = NaN(1, N);
    type_all   = strings(1, N);   % 记录每个片段的类型
     
    for i = 1:N         %parfor 并行
        sig_frag = sig_frags(i);
    
        t_offset = sig_frag.meta.t_start_s;
         
        sig_frag_dwt     = f_signal_dwt(sig_frag, "Plot", false, "wavelet", 'db4', 'level', 5);
        sig_frag_dwt_eng = f_dwt_energy(sig_frag_dwt, "HalfWin", 5, ...
            "Mode", "peak_window", "Plot", true, "Normalize", false);
        sig_frag_type    = f_classify_signal_type(sig_frag_dwt_eng);
         
        type_all(i) = sig_frag_type;
         
        if sig_frag_type == "scan"
             try
                fit_paras     = f_gauss_fit(sig_frag, "Algorithm", "lsqcurvefit", "Plot", false);
                t_scan_all(i) = fit_paras.mu + t_offset;
            catch ME
                warning('片段 %d（扫描光）拟合失败：%s', i, ME.message);
                % 保持 NaN
            end
        elseif sig_frag_type == "sync"
            try
                fit_paras     = f_fit_sync_pulse(sig_frag, "Vbase0", 0, "Plot", false);
                t_sync_all(i) = fit_paras.t0 + t_offset;
            catch ME
                warning('片段 %d（同步光）拟合失败：%s', i, ME.message);
            end
        end
    end
     
    % 提取有效结果（只保留对应类型的片段）
    t_scan = t_scan_all(type_all == "scan");
    t_sync = t_sync_all(type_all == "sync");

    
    t_scan_grop = f_pulse_grouping(t_scan,'Plot',false,'PeriodTol',0.0005,'PhaseTol',0.01,'RPM_Min',1500,'RPM_Max',3000);     %多站分类
	t_sync_grop = f_pulse_grouping(t_sync,'Plot',false,'PeriodTol',0.0005,'PhaseTol',0.01,'RPM_Min',1500,'RPM_Max',3000);
    
    stations = f_pulse_matching(t_scan_grop,t_sync_grop,'PeriodTol',0.0005,'Verbose',false);
    
    
    fprintf('同步光：%d 个有效 / %d 个片段；扫描光：%d 个有效 / %d 个片段。\n', ...
        sum(~isnan(t_sync)), numel(t_sync), sum(~isnan(t_scan)), numel(t_scan));
    

end





