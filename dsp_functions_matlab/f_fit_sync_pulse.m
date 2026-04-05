function result = f_fit_sync_pulse(sig, options)
% F_FIT_SYNC_PULSE  对 wMPS 同步脉冲进行参数拟合
%
% 拟合模型：
%
%              / V_base + P_max*(1-exp(-(t-t0)/tau_r))/(1-exp(-Tr/tau_r)),
%              |                                          t0 < t < t0+Tr
%   V(t)  =   |
%              | V_base + P_max*exp(-(t-t0-Tr)/tau_f),   t >= t0+Tr
%              |
%              \ V_base,                                  t <= t0
%
%   t0    — 脉冲起始时刻（上升沿开始），Signal.t 坐标系 (s)
%   Tr    — 上升阶段持续时长 (s)，典型值 ~0.1 µs
%   tau_r — 上升段时间常数 (s)
%   tau_f — 下降段时间常数 (s)，典型值 ~0.5 µs
%   P_max — 峰值幅度（相对基线）(V)
%   V_base— 直流基线 (V)
%
% ── 语法 ──────────────────────────────────────────────────────────────────
%   result = f_fit_sync_pulse(sig)
%   result = f_fit_sync_pulse(sig, 'Plot', true)
%   result = f_fit_sync_pulse(sig, 'Tr0', 0.12e-6, 'tau_f0', 0.4e-6)
%
% ── 必须输入 ──────────────────────────────────────────────────────────────
%   sig        — Signal 对象，应已截取为单段同步脉冲
%
% ── 可选具名参数（初始值，NaN 表示自动估计） ─────────────────────────────
%   't0_0'     t0 初始值 (s)        默认：自动（上升沿 10% 处）
%   'Pmax0'    P_max 初始值 (V)     默认：自动（峰值 - 基线）
%   'Tr0'      Tr 初始值 (s)        默认：0.1e-6
%   'tau_r0'   tau_r 初始值 (s)     默认：Tr0 / 3
%   'tau_f0'   tau_f 初始值 (s)     默认：0.5e-6
%   'Vbase0'   基线初始值 (V)       默认：自动（信号前段均值）
%
% ── 可选具名参数（拟合控制） ──────────────────────────────────────────────
%   'Algorithm' lsqcurvefit 优化算法                默认：'trust-region-reflective'
%               可选：'trust-region-reflective'（支持上下界约束）
%                     'levenberg-marquardt'（不使用上下界，忽略 lb/ub）
%   'Plot'     是否绘制拟合结果     默认：false
%   'Verbose'  是否打印参数摘要     默认：false
%
% ── 输出结构体字段 ────────────────────────────────────────────────────────
%   .t0        脉冲起始时刻 (s)，Signal.t 坐标系
%   .Tr        上升阶段持续时长 (s)
%   .tau_r     上升时间常数 (s)
%   .tau_f     下降时间常数 (s)
%   .Pmax      峰值幅度（相对基线）(V)
%   .Vbase     直流基线 (V)
%   .t_peak    理论峰值时刻 = t0 + Tr (s)，Signal.t 坐标系
%   .residual  拟合残差 RMS (V)
%   .exitflag  lsqcurvefit 退出标志
%   .fit_t     拟合所用时间轴 (s)（= sig.t，便于对比绘图）
%   .fit_y     对应的拟合曲线值 (V)

    %% ── 参数声明 ─────────────────────────────────────────────────────────
    arguments
        sig                         Signal
        options.t0_0      (1,1) double = NaN
        options.Pmax0     (1,1) double = NaN
        options.Tr0       (1,1) double = 0.1e-6
        options.tau_r0    (1,1) double = NaN        % 默认 = Tr0/3
        options.tau_f0    (1,1) double = 0.5e-6
        options.Vbase0    (1,1) double = NaN
        options.Algorithm {mustBeMember(options.Algorithm, ...
                          {'trust-region-reflective','levenberg-marquardt'})} ...
                          = 'trust-region-reflective'
        options.Plot      (1,1) logical = false
        options.Verbose   (1,1) logical = false
    end

    %% ── 提取数据 ─────────────────────────────────────────────────────────
    t = sig.t;          % 时间轴 (s)，行向量
    v = sig.data;       % 电压序列 (V)，行向量

    %% ── 自动估计初始值 ───────────────────────────────────────────────────

    % --- 基线：取信号前 5%（至少 10 点，至多 100 点）的均值 ---
    n_base  = min(max(round(sig.N * 0.05), 10), 100);
    Vbase0  = options.Vbase0;
    if isnan(Vbase0)
        Vbase0 = mean(v(1 : n_base));
    end
    v_nb = v - Vbase0;                  % 去基线

    % --- 峰值幅度 ---
    [v_pk, idx_pk] = max(v_nb);
    Pmax0 = options.Pmax0;
    if isnan(Pmax0)
        Pmax0 = max(v_pk, eps);         % 防止为零
    end

    % --- t0：默认取峰值时刻减 Tr0 ---
    t0_0 = options.t0_0;
    if isnan(t0_0)
        t0_0 = t(idx_pk) - options.Tr0;
    end
    % --- tau_r0 ---
    tau_r0 = options.tau_r0;
    if isnan(tau_r0)
        tau_r0 = options.Tr0 / 3;
    end
    % 保证 tau_r < Tr，否则模型分母接近 0
    tau_r0 = min(tau_r0, options.Tr0 * 0.9);

    %% ── 拟合参数打包 ─────────────────────────────────────────────────────
    % p = [t0, Tr, tau_r, tau_f, Pmax, Vbase]
    p0 = [t0_0, options.Tr0, tau_r0, options.tau_f0, Pmax0, Vbase0];

    % 参数下界与上界
    dt = sig.dt;
    lb = [t(1) - dt,   dt,        dt,       dt,       0,            -Inf];
    ub = [t(end),      10e-6,     5e-6,     50e-6,    Inf,           Inf];

    % 确保初始值在边界内
    p0 = max(p0, lb + eps);
    p0 = min(p0, ub - eps);

    %% ── 模型函数 ─────────────────────────────────────────────────────────
    % 分段计算，避免向量化时 tau_r≈0 导致 Inf/NaN
    model = @(p, t_vec) pulse_model(p, t_vec);

    %% ── lsqcurvefit 拟合 ─────────────────────────────────────────────────
    lsq_opts = optimoptions('lsqcurvefit', ...
        'Algorithm',              options.Algorithm, ...
        'Display',                'off', ...
        'MaxFunctionEvaluations', 5000, ...
        'MaxIterations',          2000, ...
        'FunctionTolerance',      1e-12, ...
        'StepTolerance',          1e-14, ...
        'OptimalityTolerance',    1e-12);

    % levenberg-marquardt 不支持上下界，trust-region-reflective 支持
    if strcmp(options.Algorithm, 'levenberg-marquardt')
        [p_fit, ~, residuals, exitflag] = ...
            lsqcurvefit(model, p0, t, v, [], [], lsq_opts);
    else
        [p_fit, ~, residuals, exitflag] = ...
            lsqcurvefit(model, p0, t, v, lb, ub, lsq_opts);
    end

    %% ── 整理输出 ─────────────────────────────────────────────────────────
    t0_fit    = p_fit(1);
    Tr_fit    = p_fit(2);
    tau_r_fit = p_fit(3);
    tau_f_fit = p_fit(4);
    Pmax_fit  = p_fit(5);
    Vbase_fit = p_fit(6);

    v_fitted  = model(p_fit, t);
    rms_res   = sqrt(mean(residuals .^ 2));

    result.t0       = t0_fit;
    result.Tr       = Tr_fit;
    result.tau_r    = tau_r_fit;
    result.tau_f    = tau_f_fit;
    result.Pmax     = Pmax_fit;
    result.Vbase    = Vbase_fit;
    result.t_peak   = t0_fit + Tr_fit;     % 理论峰值时刻，Signal.t 坐标系
    result.residual = rms_res;
    result.exitflag = exitflag;
    result.fit_t    = t;
    result.fit_y    = v_fitted;

    %% ── 可选：打印摘要 ───────────────────────────────────────────────────
    if options.Verbose
        fprintf('\n===== 同步脉冲拟合结果 =====\n');
        fprintf('  拟合方法        : lsqcurvefit (%s)\n', options.Algorithm);
        fprintf('  exitflag        : %d\n',        exitflag);
        fprintf('  残差 RMS        : %.4g V\n',    rms_res);
        fprintf('  ── 拟合参数 ──\n');
        fprintf('  t0   (起始时刻) : %+.6g s   (%.4g µs)\n', t0_fit,    t0_fit*1e6);
        fprintf('  Tr   (上升时长) : %.4g s     (%.4g µs)\n', Tr_fit,    Tr_fit*1e6);
        fprintf('  tau_r(上升常数) : %.4g s     (%.4g µs)\n', tau_r_fit, tau_r_fit*1e6);
        fprintf('  tau_f(下降常数) : %.4g s     (%.4g µs)\n', tau_f_fit, tau_f_fit*1e6);
        fprintf('  Pmax (峰值幅度) : %.4g V\n',   Pmax_fit);
        fprintf('  Vbase(直流基线) : %.4g V\n',   Vbase_fit);
        fprintf('  t_peak(峰值时刻): %+.6g s   (%.4g µs)\n', result.t_peak, result.t_peak*1e6);
        fprintf('============================\n\n');
    end

    %% ── 可选：绘图 ───────────────────────────────────────────────────────
    if options.Plot
        t_us     = t * 1e6;
        t_dense  = linspace(t(1), t(end), max(sig.N * 4, 2000));
        v_dense  = model(p_fit, t_dense);

        figure('Name', '同步脉冲拟合', 'NumberTitle', 'off', ...
               'Position', [100 100 800 480]);

        % 主图：原始信号 + 拟合曲线
        subplot(3,1,1:2);
        plot(t_us, v, 'Color', [0.55 0.55 0.55], 'LineWidth', 0.8, ...
             'DisplayName', '原始信号');
        hold on;
        plot(t_dense*1e6, v_dense, 'r-', 'LineWidth', 2, ...
             'DisplayName', '拟合曲线');
        xline(t0_fit*1e6,           'b--', 'LineWidth', 1.2, ...
              'Label', 't_0', 'LabelVerticalAlignment', 'bottom');
        xline(result.t_peak*1e6,    'g--', 'LineWidth', 1.2, ...
              'Label', 't_{peak}', 'LabelVerticalAlignment', 'bottom');
        ylabel('电压 (V)');
        legend('Location', 'northeast');
        grid on; box on;
        title(sprintf( ...
            ['同步脉冲拟合   t_0=%.3g µs  T_r=%.3g µs  ' ...
             '\\tau_r=%.3g µs  \\tau_f=%.3g µs  RMS=%.3g V'], ...
            t0_fit*1e6, Tr_fit*1e6, tau_r_fit*1e6, tau_f_fit*1e6, rms_res), ...
            'FontSize', 10);

        % 残差图
        subplot(3,1,3);
        stem(t_us, residuals, 'Marker', 'none', 'Color', [0.2 0.5 0.9], ...
             'LineWidth', 0.6);
        yline(0, 'k-');
        xlabel('时间 (µs)');
        ylabel('残差 (V)');
        grid on; box on;
        title(sprintf('残差  RMS = %.4g V', rms_res));
    end

end

%% ── 脉冲模型（独立子函数，避免匿名函数开销） ─────────────────────────────
function v = pulse_model(p, t)
% PULSE_MODEL  计算分段指数脉冲模型的电压值
%   p = [t0, Tr, tau_r, tau_f, Pmax, Vbase]

    t0    = p(1);
    Tr    = p(2);
    tau_r = p(3);
    tau_f = p(4);
    Pmax  = p(5);
    Vbase = p(6);

    v = zeros(size(t));

    % ── 上升段：t0 < t < t0+Tr ───────────────────────────────────────────
    m_r = (t > t0) & (t < t0 + Tr);
    if any(m_r)
        denom = 1 - exp(-Tr / tau_r);
        if denom < 1e-15          % tau_r 远大于 Tr 时退化为线性
            denom = 1e-15;
        end
        v(m_r) = Vbase + Pmax .* (1 - exp(-(t(m_r) - t0) ./ tau_r)) ./ denom;
    end

    % ── 下降段：t >= t0+Tr ───────────────────────────────────────────────
    m_f = (t >= t0 + Tr);
    if any(m_f)
        v(m_f) = Vbase + Pmax .* exp(-(t(m_f) - t0 - Tr) ./ tau_f);
    end

    % t <= t0 时保持 Vbase（已初始化为 0，加基线）
    v(t <= t0) = Vbase;
end