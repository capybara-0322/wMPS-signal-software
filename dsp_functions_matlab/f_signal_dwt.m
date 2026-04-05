function result = f_signal_dwt(sig, options)
% F_SIGNAL_DWT  对 Signal 对象执行离散小波变换（DWT）多层分解
%
% 返回值:
%   result — 结构体，包含各层系数及元信息（见下方字段说明）
%
% 用法（所有具名参数均可选，未传入时使用默认值）:
%   result = f_signal_dwt(sig)
%   result = f_signal_dwt(sig, 'wavelet', 'db4', 'level', 5)
%   result = f_signal_dwt(sig, 'level', 0, 'Plot', true)   % level=0: 自动
%
% ── 输入 ────────────────────────────────────────────────────
%   sig          Signal 对象（必须）
%
% ── 可配置选项 ───────────────────────────────────────────────
%   'wavelet'    小波函数名 (char)              默认 'db4'
%   'level'      分解层次；0 = 自动最大层数    默认 0
%   'Plot'       是否绘制各层系数图 (logical)  默认 false
%
% ── 输出结构体字段 ───────────────────────────────────────────
%   result.wavelet      小波函数名
%   result.level        实际分解层次 L
%   result.fs           原始采样率 (Hz)
%
%   result.cA{k}        第 k 层近似系数向量  (k = 1 … L)
%   result.cD{k}        第 k 层细节系数向量  (k = 1 … L)
%   result.fs_k(k)      第 k 层等效采样率 = fs / 2^k  (Hz)
%   result.t_k{k}       第 k 层对应时间轴 (s)
%   result.freq_band{k} 第 k 层细节系数对应的频率范围 [f_lo, f_hi] (Hz)
%                         cD{k}  ↔  [fs/2^(k+1),  fs/2^k]
%                         cA{L}  ↔  [0,            fs/2^(L+1)] (近似)
%
% ── 依赖 ────────────────────────────────────────────────────
%   MATLAB Wavelet Toolbox（wavedec / wrcoef / wmaxlev）

    %% ── 参数声明 ──────────────────────────────────────────────────────────
    arguments
        sig         (1,1) Signal
        options.wavelet   {mustBeText}    = 'db4'
        options.level     (1,1) double    = 0       % 0 = 自动
        options.Plot      (1,1) logical   = false
    end

    %% ── 检查 Wavelet Toolbox ─────────────────────────────────────────────
    if ~license('test', 'Wavelet_Toolbox')
        error('f_signal_dwt:noToolbox', ...
            'Wavelet Toolbox 未授权，无法执行 DWT。');
    end

    %% ── 解包 ─────────────────────────────────────────────────────────────
    wname  = char(options.wavelet);
    x      = sig.data;      % 行向量
    fs     = sig.fs;

    %% ── 确定分解层次 ─────────────────────────────────────────────────────
    max_lev = wmaxlev(numel(x), wname);

    if options.level == 0
        L = max_lev;
    else
        L = round(options.level);
        if L < 1
            error('f_signal_dwt:badLevel', 'level 必须 >= 1（或 0 表示自动）。');
        end
        if L > max_lev
            warning('f_signal_dwt:levelExceedsMax', ...
                '请求层次 %d 超过该信号/小波的最大层数 %d，已自动截断。', ...
                L, max_lev);
            L = max_lev;
        end
    end

    %% ── 多层 DWT 分解 ────────────────────────────────────────────────────
    % wavedec 返回：C（所有系数拼接行向量）和 L_vec（各段长度）
    [C, L_vec] = wavedec(x, L, wname);

    %% ── 提取各层系数 ─────────────────────────────────────────────────────
    cA = cell(1, L);
    cD = cell(1, L);

    % wavedec 的 C 结构：[cA_L | cD_L | cD_(L-1) | ... | cD_1]
    % L_vec(1)   = numel(cA_L)
    % L_vec(k+1) = numel(cD_(L+1-k))  for k = 1…L
    % L_vec(L+2) = numel(x)  （原始信号长度）

    % 最终层近似系数
    cA{L} = appcoef(C, L_vec, wname, L);

    % 各层细节系数；同时用重构得到各中间层近似系数
    for k = 1 : L
        cD{k} = detcoef(C, L_vec, k);
    end
    for k = 1 : L - 1
        cA{k} = appcoef(C, L_vec, wname, k);
    end

    %% ── 计算各层等效采样率和时间轴 ──────────────────────────────────────
    fs_k      = zeros(1, L);
    t_k       = cell(1, L);
    freq_band = cell(1, L);

    for k = 1 : L
        fs_k(k) = fs / 2^k;
        N_k     = numel(cD{k});
        t_k{k}  = (0 : N_k - 1) / fs_k(k);        % 以秒为单位

        f_hi = fs / 2^k;                            % cD{k} 上截止频率
        f_lo = fs / 2^(k+1);                        % cD{k} 下截止频率
        freq_band{k} = [f_lo, f_hi];
    end

    %% ── 打包结果 ─────────────────────────────────────────────────────────
    result = struct( ...
        'wavelet',    wname,      ...
        'level',      L,          ...
        'fs',         fs,         ...
        'cA',         {cA},       ...
        'cD',         {cD},       ...
        'fs_k',       fs_k,       ...
        't_k',        {t_k},      ...
        'freq_band',  {freq_band} ...
    );

    %% ── 可选绘图 ─────────────────────────────────────────────────────────
    if options.Plot
        plot_dwt_(result, sig);
    end

end


%% ════════════════════════════════════════════════════════════════════════
%  私有绘图辅助
%% ════════════════════════════════════════════════════════════════════════
function plot_dwt_(r, sig)
% 绘制：原始信号 + 各层 cD + 最终层 cA，共 L+2 行子图

    L      = r.level;
    n_rows = L + 2;     % 原始 + L层cD + cA_L

    fig = figure('Name', sprintf('DWT  %s  L=%d', r.wavelet, L), ...
                 'NumberTitle', 'off', ...
                 'Position', [80, 80, 1000, 120 * n_rows]);

    %── 第 1 行：原始信号 ────────────────────────────────────────────────
    ax0 = subplot(n_rows, 1, 1);
    plot(sig.t * 1e3, sig.data, 'Color', [0.2 0.2 0.2], 'LineWidth', 0.7);
    ylabel('原始');
    title(sprintf('DWT 分解  |  小波: %s  |  层数: %d  |  fs = %.3g MHz', ...
        r.wavelet, L, r.fs/1e6), 'FontWeight', 'bold');
    grid on; box on;
    set(ax0, 'XTickLabel', []);

    %── 第 2 … L+1 行：各层 cD（从第 1 层到第 L 层）──────────────────────
    colors = lines(L);
    for k = 1 : L
        ax = subplot(n_rows, 1, k + 1);

        t_ms = r.t_k{k} * 1e3;
        plot(t_ms, r.cD{k}, 'Color', colors(k,:), 'LineWidth', 0.7);

        f_lo = r.freq_band{k}(1);
        f_hi = r.freq_band{k}(2);
        ylabel(sprintf('cD%d', k));
        title(sprintf('第 %d 层细节  |  频带 [%.4g, %.4g] Hz  |  fs_k = %.4g Hz  |  N = %d', ...
            k, f_lo, f_hi, r.fs_k(k), numel(r.cD{k})), ...
            'FontSize', 8, 'FontWeight', 'normal');
        grid on; box on;
        if k < L
            set(ax, 'XTickLabel', []);
        end
    end

    %── 最后一行：最终层近似 cA_L ─────────────────────────────────────────
    subplot(n_rows, 1, n_rows);
    t_ms = r.t_k{L} * 1e3;     % cA_L 与 cD_L 长度相近，共用时间轴
    t_cA = (0 : numel(r.cA{L}) - 1) / r.fs_k(L) * 1e3;
    plot(t_cA, r.cA{L}, 'Color', [0.6 0 0.8], 'LineWidth', 0.8);
    ylabel(sprintf('cA%d', L));
    title(sprintf('第 %d 层近似  |  频带 [0, %.4g] Hz  |  fs_k = %.4g Hz  |  N = %d', ...
        L, r.freq_band{L}(1), r.fs_k(L), numel(r.cA{L})), ...
        'FontSize', 8, 'FontWeight', 'normal');
    xlabel('时间 (ms)');
    grid on; box on;

    % 所有子图共享 x 轴缩放
    linkaxes(findall(fig, 'Type', 'axes'), 'x');
end
