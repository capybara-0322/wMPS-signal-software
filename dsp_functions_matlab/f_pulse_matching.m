function stations = f_pulse_matching(t_scan_group, t_sync_group, varargin)
% f_pulse_matching  将扫描脉冲与同步脉冲按周期匹配到发射站，并计算归一化时刻
%
% 用法:
%   stations = f_pulse_matching(t_scan_group, t_sync_group)
%   stations = f_pulse_matching(t_scan_group, t_sync_group, 'Name', Value, ...)
%
% 输入:
%   t_scan_group  - f_pulse_grouping 对扫描脉冲的输出结构体
%   t_sync_group  - f_pulse_grouping 对同步脉冲的输出结构体
%
% 可选参数 (Name-Value):
%   'PeriodTol'   - 判定两组脉冲属于同一发射站的周期容差 (秒)，默认 0.5
%   'Verbose'     - 是否打印信息, true/false, 默认 false
%
% 输出:
%   stations      - 结构体数组 (按转速从小到大排列)，每个元素包含:
%     .period       - 发射站周期 (秒)
%     .rpm          - 发射站转速 (rpm)
%     .scan_groups  - 匹配到的扫描脉冲组编号 (来自 t_scan_group)
%     .sync_group   - 匹配到的同步脉冲组编号 (来自 t_sync_group)
%     .n_cycles     - 有效周期数
%     .t1_all       - 各周期的 t1' 值 (n_cycles×1)
%     .t2_all       - 各周期的 t2' 值 (n_cycles×1)
%     .t1_mean      - t1' 的平均值
%     .t2_mean      - t2' 的平均值
%     .t1_std       - t1' 的标准差
%     .t2_std       - t2' 的标准差
%     .cycles       - 各周期详细信息 (n_cycles×1 结构体数组)
%                     每个元素: .t01, .ts1, .ts2, .t02, .t1_norm, .t2_norm
%
% 说明:
%   每个发射站在一个周期内的时序为:
%     t01 (同步脉冲) → ts1 (扫描脉冲1) → ts2 (扫描脉冲2) → t02 (同步脉冲)
%   归一化时刻:
%     t1' = (ts1 - t01) / (t02 - t01)
%     t2' = (ts2 - t01) / (t02 - t01)

%% ==================== 解析参数 ====================
p = inputParser;
addRequired(p, 't_scan_group', @isstruct);
addRequired(p, 't_sync_group', @isstruct);
addParameter(p, 'PeriodTol', 0.5, @isscalar);
addParameter(p, 'Verbose', false, @islogical);
parse(p, t_scan_group, t_sync_group, varargin{:});

period_tol = p.Results.PeriodTol;
verbose    = p.Results.Verbose;

%% ==================== 提取各组信息 ====================
% 扫描脉冲: 提取每个组的周期和脉冲时刻
scan_n_groups = size(t_scan_group.summary, 1);
scan_info = struct('group_id', {}, 'period', {}, 'pulses', {});
for g = 1:scan_n_groups
    gid = t_scan_group.summary(g, 1);
    T   = t_scan_group.summary(g, 2);
    idx = (t_scan_group.group_id == gid);
    scan_info(end+1).group_id = gid;
    scan_info(end).period     = T;
    scan_info(end).pulses     = sort(t_scan_group.pulses(idx));
end

% 同步脉冲: 提取每个组的周期和脉冲时刻
sync_n_groups = size(t_sync_group.summary, 1);
sync_info = struct('group_id', {}, 'period', {}, 'pulses', {});
for g = 1:sync_n_groups
    gid = t_sync_group.summary(g, 1);
    T   = t_sync_group.summary(g, 2);
    idx = (t_sync_group.group_id == gid);
    sync_info(end+1).group_id = gid;
    sync_info(end).period     = T;
    sync_info(end).pulses     = sort(t_sync_group.pulses(idx));
end

if verbose
    fprintf('=== f_pulse_matching ===\n');
    fprintf('扫描脉冲组数: %d\n', scan_n_groups);
    for i = 1:length(scan_info)
        fprintf('  扫描组 %d: T=%.4f, %d 个脉冲\n', ...
                scan_info(i).group_id, scan_info(i).period, length(scan_info(i).pulses));
    end
    fprintf('同步脉冲组数: %d\n', sync_n_groups);
    for i = 1:length(sync_info)
        fprintf('  同步组 %d: T=%.4f, %d 个脉冲\n', ...
                sync_info(i).group_id, sync_info(i).period, length(sync_info(i).pulses));
    end
end

%% ==================== 按周期匹配到发射站 ====================
% 收集所有组(扫描+同步)的周期，做聚类
% 每个条目: [周期, 类型(1=scan,2=sync), 在scan_info/sync_info中的索引]
all_entries = [];
for i = 1:length(scan_info)
    all_entries = [all_entries; scan_info(i).period, 1, i];
end
for i = 1:length(sync_info)
    all_entries = [all_entries; sync_info(i).period, 2, i];
end

if isempty(all_entries)
    stations = [];
    if verbose
        fprintf('没有找到任何脉冲组，返回空结果。\n');
    end
    return;
end

% 按周期排序
all_entries = sortrows(all_entries, 1);

% 单链接聚类: 相邻周期之差 < period_tol 则归为同一发射站
station_labels = ones(size(all_entries, 1), 1);
sid = 1;
for i = 2:size(all_entries, 1)
    if all_entries(i, 1) - all_entries(i-1, 1) > period_tol
        sid = sid + 1;
    end
    station_labels(i) = sid;
end
n_stations = max(station_labels);

if verbose
    fprintf('\n按周期聚类得到 %d 个发射站 (PeriodTol=%.3f):\n', n_stations, period_tol);
end

%% ==================== 逐发射站处理 ====================
stations_raw = struct([]);

for s = 1:n_stations
    mask = (station_labels == s);
    entries = all_entries(mask, :);
    
    % 分拣扫描组和同步组
    scan_idx  = entries(entries(:,2) == 1, 3);  % scan_info 中的索引
    sync_idx  = entries(entries(:,2) == 2, 3);  % sync_info 中的索引
    
    % 该发射站的平均周期
    mean_period = mean(entries(:, 1));
    
    if verbose
        fprintf('\n--- 发射站 %d (平均周期=%.4f, 转速=%.2f rpm) ---\n', ...
                s, mean_period, 60/mean_period);
        fprintf('  扫描组: %d 个, 同步组: %d 个\n', length(scan_idx), length(sync_idx));
    end
    
    % 检查: 应有2组扫描、1组同步
    if length(sync_idx) < 1
        if verbose
            fprintf('  警告: 无同步脉冲组，跳过该发射站。\n');
        end
        continue;
    end
    if length(scan_idx) < 2
        if verbose
            fprintf('  警告: 扫描脉冲组不足2个（实际%d个），跳过该发射站。\n', length(scan_idx));
        end
        continue;
    end
    
    % 取第一个同步组（如果有多个同步组取脉冲最多的）
    if length(sync_idx) == 1
        sync_sel = sync_idx(1);
    else
        sync_counts = zeros(length(sync_idx), 1);
        for i = 1:length(sync_idx)
            sync_counts(i) = length(sync_info(sync_idx(i)).pulses);
        end
        [~, best] = max(sync_counts);
        sync_sel = sync_idx(best);
        if verbose
            fprintf('  注意: 有 %d 个同步组，选取脉冲最多的组 %d。\n', ...
                    length(sync_idx), sync_info(sync_sel).group_id);
        end
    end
    
    % 取两个扫描组（如果超过2个，取脉冲最多的两个）
    if length(scan_idx) == 2
        scan_sel = scan_idx;
    else
        scan_counts = zeros(length(scan_idx), 1);
        for i = 1:length(scan_idx)
            scan_counts(i) = length(scan_info(scan_idx(i)).pulses);
        end
        [~, order] = sort(scan_counts, 'descend');
        scan_sel = scan_idx(order(1:2));
        if verbose
            fprintf('  注意: 有 %d 个扫描组，选取脉冲最多的两个组。\n', length(scan_idx));
        end
    end
    
    % 获取同步脉冲和两组扫描脉冲
    t_sync  = sort(sync_info(sync_sel).pulses);
    t_scanA = sort(scan_info(scan_sel(1)).pulses);
    t_scanB = sort(scan_info(scan_sel(2)).pulses);
    
    if verbose
        fprintf('  同步脉冲 (组%d): %d 个\n', sync_info(sync_sel).group_id, length(t_sync));
        fprintf('  扫描脉冲A (组%d): %d 个\n', scan_info(scan_sel(1)).group_id, length(t_scanA));
        fprintf('  扫描脉冲B (组%d): %d 个\n', scan_info(scan_sel(2)).group_id, length(t_scanB));
    end
    
    %% ====== 逐周期提取 t01, ts1, ts2, t02 ======
    % 每个周期由两个相邻同步脉冲界定: [t_sync(i), t_sync(i+1)]
    % 在此区间内找两个扫描脉冲（一个来自A，一个来自B）
    
    t1_list = [];
    t2_list = [];
    cycles  = struct([]);
    
    for i = 1:length(t_sync) - 1
        t01 = t_sync(i);
        t02 = t_sync(i+1);
        dt  = t02 - t01;
        
        % 周期应与平均周期接近，否则跳过（可能是漏脉冲导致的长间隔）
        if dt < mean_period * 0.5 || dt > mean_period * 1.5
            continue;
        end
        
        % 在 (t01, t02) 内找扫描脉冲A
        sA = t_scanA(t_scanA > t01 & t_scanA < t02);
        % 在 (t01, t02) 内找扫描脉冲B
        sB = t_scanB(t_scanB > t01 & t_scanB < t02);
        
        % 应各有恰好1个
        if length(sA) ~= 1 || length(sB) ~= 1
            continue;
        end
        
        % ts1 是较早的扫描脉冲，ts2 是较晚的
        ts1 = min(sA, sB);
        ts2 = max(sA, sB);
        
        % 计算归一化时刻
        t1_norm = (ts1 - t01) / dt;
        t2_norm = (ts2 - t01) / dt;
        
        % 基本合理性检查
        if t1_norm <= 0 || t1_norm >= 1 || t2_norm <= 0 || t2_norm >= 1 || t1_norm >= t2_norm
            continue;
        end
        
        t1_list(end+1, 1) = t1_norm;
        t2_list(end+1, 1) = t2_norm;
        
        cyc.t01     = t01;
        cyc.ts1     = ts1;
        cyc.ts2     = ts2;
        cyc.t02     = t02;
        cyc.t1_norm = t1_norm;
        cyc.t2_norm = t2_norm;
        if isempty(cycles)
            cycles = cyc;
        else
            cycles(end+1) = cyc;
        end
    end
    
    n_cycles = length(t1_list);
    
    if n_cycles == 0
        if verbose
            fprintf('  警告: 无有效周期，跳过该发射站。\n');
        end
        continue;
    end
    
    % 组装该发射站结果
    st.period      = mean_period;
    st.rpm         = 60 / mean_period;
    st.scan_groups = [scan_info(scan_sel(1)).group_id, scan_info(scan_sel(2)).group_id];
    st.sync_group  = sync_info(sync_sel).group_id;
    st.n_cycles    = n_cycles;
    st.t1_all      = t1_list;
    st.t2_all      = t2_list;
    st.t1_mean     = mean(t1_list);
    st.t2_mean     = mean(t2_list);
    st.t1_std      = std(t1_list);
    st.t2_std      = std(t2_list);
    st.cycles      = cycles;
    
    if isempty(stations_raw)
        stations_raw = st;
    else
        stations_raw(end+1) = st;
    end
    
    if verbose
        fprintf('  有效周期数: %d\n', n_cycles);
        fprintf('  t1'' 平均值: %.6f (std=%.6f)\n', st.t1_mean, st.t1_std);
        fprintf('  t2'' 平均值: %.6f (std=%.6f)\n', st.t2_mean, st.t2_std);
    end
end

%% ==================== 按转速从小到大排列 ====================
if isempty(stations_raw)
    stations = [];
    if verbose
        fprintf('\n未找到任何有效发射站。\n');
    end
    return;
end

rpms = [stations_raw.rpm];
[~, sort_idx] = sort(rpms, 'ascend');
stations = stations_raw(sort_idx);

if verbose
    fprintf('\n==================== 最终结果 ====================\n');
    fprintf('共 %d 个发射站 (按转速升序排列):\n', length(stations));
    fprintf('  %6s  %10s  %10s  %8s  %10s  %10s\n', ...
            '编号', '周期(s)', '转速(rpm)', '周期数', 't1''均值', 't2''均值');
    for i = 1:length(stations)
        fprintf('  %6d  %10.4f  %10.2f  %8d  %10.6f  %10.6f\n', ...
                i, stations(i).period, stations(i).rpm, ...
                stations(i).n_cycles, stations(i).t1_mean, stations(i).t2_mean);
    end
end

end
