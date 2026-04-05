function results = f_pulse_grouping(pulses, varargin)
% f_pulse_grouping  基于两两时间差聚类的周期脉冲分组
%
% 用法:
%   results = f_pulse_grouping(pulses)
%   results = f_pulse_grouping(pulses, 'Name', Value, ...)
%
% 输入:
%   pulses      - 脉冲时刻数组 (1×N 或 N×1)
%
% 可选参数 (Name-Value):
%   'PeriodTol'     - 周期聚类阈值，判定两个时间差属于同一周期的容忍度
%                     默认 0.3（绝对值）
%   'PhaseTol'      - 相位聚类阈值，同一周期组内判定同相位的容忍度
%                     默认 0.2（归一化到周期，即 0.2*T）
%   'MinCount'      - 一个周期簇至少包含多少个时间差才认为有效
%                     默认 5
%   'MinPulses'     - 一个最终分组至少包含多少个脉冲才保留
%                     默认 3
%   'Plot'          - 是否画图, true/false, 默认 false
%   'Verbose'       - 是否在命令行打印信息, true/false, 默认 false
%   'RPM_Min'       - 转速下限 (rpm)，用于计算周期上限，默认 [] (不限制)
%   'RPM_Max'       - 转速上限 (rpm)，用于计算周期下限，默认 [] (不限制)
%                     周期 = 60/rpm (秒)，仅保留周期在 [60/RPM_Max, 60/RPM_Min] 内的簇
%
% 输出:
%   results         - 结构体，包含:
%     .pulses         - 排序后的脉冲时刻
%     .group_id       - 每个脉冲的最终分组编号 (0=未分配)
%     .periods        - 每个分组对应的周期
%     .phases         - 每个分组对应的相位
%     .freq_labels    - 每个脉冲的频率组编号（相位细分前）
%     .summary        - 各组的汇总表
%
% 示例:
%   t = sort([0.3 + 5*(0:19), 2.1 + 7.5*(0:14), 1.0 + 12*(0:9)] + 0.05*randn(1,45));
%   results = f_pulse_grouping(t, 'PeriodTol', 0.3, 'PhaseTol', 0.2, 'Plot', true);

%% ==================== 解析参数 ====================
p = inputParser;
addRequired(p, 'pulses', @isnumeric);
addParameter(p, 'PeriodTol', 0.3, @isscalar);
addParameter(p, 'PhaseTol', 0.2, @isscalar);
addParameter(p, 'MinCount', 5, @isscalar);
addParameter(p, 'MinPulses', 3, @isscalar);
addParameter(p, 'Plot', false, @islogical);
addParameter(p, 'Verbose', false, @islogical);
addParameter(p, 'RPM_Min', [], @(x) isempty(x) || isscalar(x));
addParameter(p, 'RPM_Max', [], @(x) isempty(x) || isscalar(x));
parse(p, pulses, varargin{:});

period_tol  = p.Results.PeriodTol;
phase_tol   = p.Results.PhaseTol;   % 归一化到周期的比例
min_count   = p.Results.MinCount;
min_pulses  = p.Results.MinPulses;
do_plot     = p.Results.Plot;
verbose     = p.Results.Verbose;

% 由转速上下限计算周期上下限: T = 60 / rpm
rpm_min = p.Results.RPM_Min;
rpm_max = p.Results.RPM_Max;

if ~isempty(rpm_max) && rpm_max > 0
    T_min = 60 / rpm_max;
else
    T_min = 0;
end

if ~isempty(rpm_min) && rpm_min > 0
    T_max = 60 / rpm_min;
else
    T_max = Inf;
end

%% ==================== 预处理 ====================
pulses = pulses(:);
pulses = sort(pulses);
n = length(pulses);

if verbose
    fprintf('=== f_pulse_grouping ===\n');
    fprintf('脉冲数量: %d\n', n);
    fprintf('参数: PeriodTol=%.3f, PhaseTol=%.3f, MinCount=%d, MinPulses=%d\n', ...
            period_tol, phase_tol, min_count, min_pulses);
    if T_min > 0 || isfinite(T_max)
        fprintf('转速范围: [');
        if ~isempty(rpm_min), fprintf('%.1f', rpm_min); else, fprintf('-'); end
        fprintf(', ');
        if ~isempty(rpm_max), fprintf('%.1f', rpm_max); else, fprintf('-'); end
        fprintf('] rpm  →  周期范围: [%.6f, ', T_min);
        if isfinite(T_max), fprintf('%.6f', T_max); else, fprintf('Inf'); end
        fprintf('] s\n');
    end
end

%% ========== 第一步: 两两做差 ==========
diffs = [];
diff_i = [];  % 记录差值来源的索引对
diff_j = [];
for i = 1:n
    for j = i+1:n
        diffs = [diffs; pulses(j) - pulses(i)];
        diff_i = [diff_i; i];
        diff_j = [diff_j; j];
    end
end
if verbose
    fprintf('\n[第一步] 两两时间差: %d 个\n', length(diffs));
end

%% ========== 第二步: 对时间差做简单聚类，找候选周期 ==========
% 思路: 对所有正时间差排序后，用阈值 period_tol 做单链接聚类
%       每个簇的中位数作为候选周期

diffs_sorted = sort(diffs);

% 单链接聚类: 相邻差值之差 < period_tol 则归为同簇
cluster_id = ones(size(diffs_sorted));
cid = 1;
for i = 2:length(diffs_sorted)
    if diffs_sorted(i) - diffs_sorted(i-1) > period_tol
        cid = cid + 1;
    end
    cluster_id(i) = cid;
end

% 统计每个簇的大小和中位数
n_clusters = max(cluster_id);
cluster_info = zeros(n_clusters, 3);  % [簇编号, 元素数, 中位数]
for c = 1:n_clusters
    mask = (cluster_id == c);
    cluster_info(c, :) = [c, sum(mask), median(diffs_sorted(mask))];
end

% 按元素数量降序排列
cluster_info = sortrows(cluster_info, -2);

% 过滤掉太小的簇
cluster_info = cluster_info(cluster_info(:,2) >= min_count, :);

% 过滤掉不在周期范围内的簇 (由 RPM_Min/RPM_Max 确定)
in_range = cluster_info(:,3) >= T_min & cluster_info(:,3) <= T_max;
cluster_info = cluster_info(in_range, :);

if verbose
    fprintf('\n[第二步] 时间差聚类结果 (共 %d 个有效簇):\n', size(cluster_info, 1));
    fprintf('  %8s  %8s  %12s\n', '簇编号', '元素数', '中位数(周期)');
    for k = 1:size(cluster_info, 1)
        fprintf('  %8d  %8d  %12.4f\n', cluster_info(k,1), cluster_info(k,2), cluster_info(k,3));
    end
end

%% ========== 第三步: 提取基本周期（去谐波） ==========
% 从元素最多的簇开始，检查是否为已有周期的整数倍或整数分之一
candidate_periods = cluster_info(:, 3);
candidate_counts  = cluster_info(:, 2);

base_periods = [];  % 最终保留的独立基本周期
base_counts  = [];

for k = 1:length(candidate_periods)
    T = candidate_periods(k);
    is_harmonic = false;
    
    for m = 1:length(base_periods)
        Tb = base_periods(m);
        ratio = T / Tb;
        % 检查是否为整数倍 (2T, 3T, ...) 或整数分之一 (T/2, T/3, ...)
        if ratio > 0.5
            nearest_int = round(ratio);
            if nearest_int >= 2 && abs(ratio - nearest_int) < period_tol / Tb
                is_harmonic = true;
                break;
            end
        end
        ratio_inv = Tb / T;
        if ratio_inv >= 2 && abs(ratio_inv - round(ratio_inv)) < period_tol / T
            % T 是 Tb 的因子，用更小的周期替换
            base_periods(m) = T;
            base_counts(m) = max(base_counts(m), candidate_counts(k));
            is_harmonic = true;
            break;
        end
    end
    
    if ~is_harmonic
        base_periods = [base_periods; T];
        base_counts  = [base_counts; candidate_counts(k)];
    end
end

% 按计数降序
[base_counts, sort_idx] = sort(base_counts, 'descend');
base_periods = base_periods(sort_idx);

if verbose
    fprintf('\n[第三步] 独立基本周期 (%d 个):\n', length(base_periods));
    for k = 1:length(base_periods)
        fprintf('  T_%d = %.4f  (支持差值数: %d)\n', k, base_periods(k), base_counts(k));
    end
end

%% ========== 第四步: 按周期给每个脉冲指定频率组 ==========
% 从脉冲最多（支持度最高）的周期开始，贪心分配
freq_labels = zeros(n, 1);   % 0 = 未分配
assigned_period = zeros(n, 1);  % 每个脉冲对应的周期值

for k = 1:length(base_periods)
    T = base_periods(k);
    unassigned = find(freq_labels == 0);
    
    if length(unassigned) < min_pulses
        break;
    end
    
    % 对未分配的脉冲，检查它与已分配到本组的脉冲（或其他未分配脉冲）
    % 的时间差是否为 T 的整数倍
    % 策略: 对所有未分配脉冲，计算 t mod T，看哪些相位聚集在一起
    
    t_un = pulses(unassigned);
    phases_mod = mod(t_un, T);
    
    % 在相位空间上找簇 (环形距离，阈值 = phase_tol * T)
    tol_abs = phase_tol * T;
    
    % 简单贪心: 不断找最大的相位簇
    remaining_mask = true(size(unassigned));
    
    while sum(remaining_mask) >= min_pulses
        rem_idx = find(remaining_mask);
        rem_phases = mod(pulses(unassigned(rem_idx)), T);
        
        % 对每个点统计邻域内有多少点
        neighbor_count = zeros(size(rem_idx));
        for i = 1:length(rem_idx)
            cdist = abs(rem_phases - rem_phases(i));
            cdist = min(cdist, T - cdist);
            neighbor_count(i) = sum(cdist < tol_abs);
        end
        
        [max_neighbors, seed_idx] = max(neighbor_count);
        if max_neighbors < min_pulses
            break;
        end
        
        seed_phase = rem_phases(seed_idx);
        cdist = abs(rem_phases - seed_phase);
        cdist = min(cdist, T - cdist);
        in_cluster = cdist < tol_abs;
        
        % 验证: 检查这些点的间隔是否真的是 T 的整数倍
        cluster_times = sort(pulses(unassigned(rem_idx(in_cluster))));
        intervals = diff(cluster_times);
        ratios = intervals / T;
        valid = abs(ratios - round(ratios)) < phase_tol;
        
        if sum(valid) >= min_pulses - 1  % 至少 min_pulses-1 个间隔合格
            cluster_global_idx = unassigned(rem_idx(in_cluster));
            freq_labels(cluster_global_idx) = k;
            assigned_period(cluster_global_idx) = T;
            remaining_mask(rem_idx(in_cluster)) = false;
        else
            % 这个种子点不行，标记跳过
            remaining_mask(rem_idx(seed_idx)) = false;
        end
    end
end

n_freq_assigned = sum(freq_labels > 0);
if verbose
    fprintf('\n[第四步] 频率分配: %d/%d 个脉冲已分配\n', n_freq_assigned, n);
    for k = 1:length(base_periods)
        fprintf('  频率组 %d (T=%.4f): %d 个脉冲\n', k, base_periods(k), sum(freq_labels == k));
    end
end

%% ========== 第五步: 在每个频率组内按相位细分 ==========
group_id = zeros(n, 1);
group_periods = [];
group_phases  = [];
gid = 0;

for k = 1:length(base_periods)
    T = base_periods(k);
    members = find(freq_labels == k);
    
    if isempty(members)
        continue;
    end
    
    % 计算组内每个脉冲的相位
    ph = mod(pulses(members), T);
    
    % 对相位做环形聚类
    assigned = false(size(members));
    
    while sum(~assigned) >= min_pulses
        rem = find(~assigned);
        rem_ph = ph(rem);
        
        % 找密度最高点
        nc = zeros(size(rem));
        tol_abs = phase_tol * T;
        for i = 1:length(rem)
            cd = abs(rem_ph - rem_ph(i));
            cd = min(cd, T - cd);
            nc(i) = sum(cd < tol_abs);
        end
        
        [mx, si] = max(nc);
        if mx < min_pulses
            break;
        end
        
        seed_ph = rem_ph(si);
        cd = abs(rem_ph - seed_ph);
        cd = min(cd, T - cd);
        in_cluster = cd < tol_abs;
        
        gid = gid + 1;
        group_id(members(rem(in_cluster))) = gid;
        assigned(rem(in_cluster)) = true;
        
        % 计算该组平均相位（圆形均值）
        cluster_ph = rem_ph(in_cluster);
        mean_phase = mod(atan2(mean(sin(2*pi*cluster_ph/T)), ...
                               mean(cos(2*pi*cluster_ph/T))) * T / (2*pi), T);
        
        group_periods = [group_periods; T];
        group_phases  = [group_phases; mean_phase];
    end
end

if verbose
    fprintf('\n[第五步] 最终分组: %d 个组\n', gid);
    fprintf('  %6s  %10s  %10s  %8s\n', '组号', '周期', '相位', '脉冲数');
end
summary = [];
for g = 1:gid
    cnt = sum(group_id == g);
    if verbose
        fprintf('  %6d  %10.4f  %10.4f  %8d\n', g, group_periods(g), group_phases(g), cnt);
    end
    summary = [summary; g, group_periods(g), group_phases(g), cnt];
end

n_unassigned = sum(group_id == 0);
if verbose && n_unassigned > 0
    fprintf('  未分配: %d 个脉冲\n', n_unassigned);
end

%% ==================== 组装输出 ====================
results.pulses      = pulses;
results.group_id    = group_id;
results.periods     = group_periods;
results.phases      = group_phases;
results.freq_labels = freq_labels;
results.base_periods = base_periods;
results.summary     = summary;
results.diffs       = diffs;

%% ==================== 可选画图 ====================
if do_plot
    f_pulse_grouping_plot(results);
end

end


%% ================== 画图子函数 ==================
function f_pulse_grouping_plot(results)

    pulses   = results.pulses;
    group_id = results.group_id;
    diffs    = results.diffs;
    base_periods = results.base_periods;
    n_groups = max(group_id);
    
    figure('Name', '脉冲分组结果', 'Position', [80 80 1300 850], 'Color', 'w');
    colors = lines(max(n_groups, 1));
    
    % --- 子图1: 时间差直方图 ---
    subplot(2,2,1);
    histogram(diffs, 300, 'FaceColor', [0.4 0.6 0.9], 'EdgeColor', 'none');
    hold on;
    for k = 1:length(base_periods)
        xline(base_periods(k), 'r-', sprintf('T=%.2f', base_periods(k)), ...
              'LineWidth', 1.5, 'FontSize', 9, 'LabelOrientation', 'horizontal');
    end
    xlabel('\Deltat'); ylabel('计数');
    title('两两时间差直方图');
    grid on; box on;
    
    % --- 子图2: 分组结果时间轴 ---
    subplot(2,2,2);
    hold on;
    for i = 1:length(pulses)
        g = group_id(i);
        if g == 0
            plot(pulses(i), 0, 'kx', 'MarkerSize', 8, 'LineWidth', 1.5);
        else
            plot(pulses(i), g, 'o', 'Color', colors(g,:), ...
                 'MarkerFaceColor', colors(g,:), 'MarkerSize', 6);
        end
    end
    xlabel('时刻 t'); ylabel('组号');
    title('脉冲分组结果');
    yticks(0:n_groups);
    grid on; box on;
    
    % --- 子图3: 各组归一化相位分布 ---
    subplot(2,2,3);
    hold on;
    legend_entries = {};
    for g = 1:n_groups
        T  = results.periods(g);
        idx = (group_id == g);
        ph = mod(pulses(idx), T) / T;  % 归一化到 [0, 1)
        plot(ph, g * ones(size(ph)), 'o', 'Color', colors(g,:), ...
             'MarkerFaceColor', colors(g,:), 'MarkerSize', 6);
        legend_entries{end+1} = sprintf('组%d (T=%.2f)', g, T);
    end
    xlabel('归一化相位 (t mod T) / T');
    ylabel('组号');
    title('各组相位分布');
    xlim([0 1]);
    yticks(1:n_groups);
    if ~isempty(legend_entries)
        legend(legend_entries, 'Location', 'bestoutside', 'FontSize', 8);
    end
    grid on; box on;
    
    % --- 子图4: 残差检验 ---
    subplot(2,2,4);
    hold on;
    for g = 1:n_groups
        T  = results.periods(g);
        ph = results.phases(g);
        idx = find(group_id == g);
        t_g = pulses(idx);
        % 残差 = 实际时刻 - 最近的理想格点
        residuals = t_g - ph - round((t_g - ph) / T) * T;
        plot(t_g, residuals, 'o', 'Color', colors(g,:), ...
             'MarkerFaceColor', colors(g,:), 'MarkerSize', 5);
    end
    yline(0, 'k--');
    xlabel('时刻 t'); ylabel('残差');
    title('各组残差 (偏离理想格点)');
    grid on; box on;
    
    sgtitle('f\_pulse\_grouping 结果', 'FontSize', 14, 'FontWeight', 'bold');
end