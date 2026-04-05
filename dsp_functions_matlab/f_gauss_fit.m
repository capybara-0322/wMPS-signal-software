function result = f_gauss_fit(sig, options)
% F_GAUSS_FIT  对 Signal 对象进行高斯函数拟合
%
% 拟合模型（由 'Baseline' 参数控制）：
%   无基线：  V(t) = A * exp( -(t - mu)^2 / (2*sigma^2) )
%   含基线：  V(t) = A * exp( -(t - mu)^2 / (2*sigma^2) ) + C
%
% 语法：
%   result = f_gauss_fit(sig)
%   result = f_gauss_fit(sig, 'Algorithm', 'lsqcurvefit')
%   result = f_gauss_fit(sig, 'x0', [A0, mu0, sigma0])
%   result = f_gauss_fit(sig, 'Baseline', true, 'x0', [A0, mu0, sigma0, C0])
%   result = f_gauss_fit(sig, 'Plot', true)
%
% ── 输入 ──────────────────────────────────────────────────────────────────
%   sig             Signal 对象（必须），始终对整段信号进行拟合
%
% ── 可选具名参数 ──────────────────────────────────────────────────────────
%   'Algorithm'     拟合算法，可选：
%                     'lsqcurvefit'  — 需 Optimization Toolbox（默认）
%                     'fit'          — 需 Curve Fitting Toolbox
%                     'fminsearch'   — 仅需 MATLAB 基础版，兜底方案
%                   指定算法对应 Toolbox 不可用时，自动降级并发出警告。
%                   降级链：lsqcurvefit/fit → fminsearch。
%
%   'x0'            初始参数向量（double 行向量，物理单位）
%                     无基线：[A (V), mu (s), sigma (s)]
%                     含基线：[A (V), mu (s), sigma (s), C (V)]
%                   未传入时使用固定默认值（见下文"默认初始值"）。
%
%   'Baseline'      是否拟合常数基线偏移 C（logical，默认 false）
%
%   'Plot'          是否绘制拟合结果对比图（logical，默认 false）
%
% ── 默认初始值（'x0' 未传入时）────────────────────────────────────────────
%   在归一化域（时间平移缩放至 [-1,1]、幅值缩放至 [-1,1]）下的固定值：
%     A0     = 1.0   （对应约 1 倍的归一化幅值峰值）
%     mu0    = 0.0   （对应归一化时间轴中点）
%     sigma0 = 0.1   （对应归一化时间轴总长的 1/10）
%     C0     = 0.0   （仅 Baseline=true 时有效，基线默认为 0）
%
% ── 输出 result（struct）─────────────────────────────────────────────────
%   result.A          峰值幅度（V）
%   result.mu         峰值时刻（s）
%   result.sigma      高斯标准差（s）
%   result.C          基线偏移（V），仅 Baseline=true 时存在
%   result.fwhm       半高全宽（s），= 2*sqrt(2*ln2)*sigma
%   result.residual   均方根残差（V）
%   result.r_squared  R² 决定系数
%   result.algorithm  实际使用的算法名（char）
%   result.converged  是否收敛（logical）
%   result.x_fit      拟合所用时间轴（s，行向量）
%   result.y_fit      拟合曲线值（V，行向量）

    %% ── 参数声明 ──────────────────────────────────────────────────────────
    arguments
        sig               Signal
        options.Algorithm    {mustBeMember(options.Algorithm, ...
                               {'lsqcurvefit','fit','fminsearch'})} = 'lsqcurvefit'
        options.x0           (1,:) double  = []
        options.Baseline     (1,1) logical = false
        options.Plot         (1,1) logical = false
    end

    t_raw = sig.t;
    y_raw = sig.data;

    use_baseline = options.Baseline;
    n_param      = 3 + use_baseline;

    %% ── 数值归一化 ────────────────────────────────────────────────────────
    % 时间轴平移使均值为 0，缩放使范围为 [-1, 1]；幅值按最大绝对值缩放。
    % 归一化后各参数量级接近 1，可有效减少拟合算法的病态性。
    t_offset = mean(t_raw);
    t_scale  = max(abs(t_raw - t_offset));
    if t_scale < eps
        error('f_gauss_fit:degenerate', '信号时间跨度过小，无法进行拟合。');
    end

    y_scale = max(abs(y_raw));
    if y_scale < eps
        error('f_gauss_fit:zeroSignal', '信号幅值近似为零，无法进行拟合。');
    end

    t_norm = (t_raw - t_offset) / t_scale;
    y_norm = y_raw / y_scale;

    %% ── 初始值处理 ────────────────────────────────────────────────────────
    if isempty(options.x0)
        % 固定默认初始值（归一化域）
        if use_baseline
            x0_norm = [1.0, 0.0, 0.1, 0.0];
        else
            x0_norm = [1.0, 0.0, 0.1];
        end
    else
        % 用户传入物理量初始值，校验维度后转换至归一化域
        if numel(options.x0) ~= n_param
            error('f_gauss_fit:x0Dim', ...
                'x0 维度应为 %d（Baseline=%d），但传入了 %d 个元素。', ...
                n_param, use_baseline, numel(options.x0));
        end
        x0_u    = options.x0(:)';
        A0_n    =  x0_u(1) / y_scale;
        mu0_n   = (x0_u(2) - t_offset) / t_scale;
        sig0_n  =  x0_u(3) / t_scale;
        if use_baseline
            C0_n    = x0_u(4) / y_scale;
            x0_norm = [A0_n, mu0_n, sig0_n, C0_n];
        else
            x0_norm = [A0_n, mu0_n, sig0_n];
        end
    end

    %% ── 定义拟合模型与参数约束 ────────────────────────────────────────────
    if use_baseline
        gauss_fn = @(p, t) p(1) .* exp(-(t - p(2)).^2 ./ (2 .* p(3).^2)) + p(4);
    else
        gauss_fn = @(p, t) p(1) .* exp(-(t - p(2)).^2 ./ (2 .* p(3).^2));
    end

    % sigma（第 3 个参数）须严格为正，其余参数无界
    lb_norm = [-inf, -inf, 1e-9, -inf * ones(1, use_baseline)];
    ub_norm = [ inf,  inf,  inf,  inf * ones(1, use_baseline)];

    cost_fn = @(p) sum((gauss_fn(p, t_norm) - y_norm).^2);

    %% ── 算法可用性检查与降级 ──────────────────────────────────────────────
    algo = options.Algorithm;

    if strcmp(algo, 'lsqcurvefit') && ~license('test', 'optimization_toolbox')
        warning('f_gauss_fit:noOptToolbox', ...
            'Optimization Toolbox 不可用，自动降级为 fminsearch。');
        algo = 'fminsearch';
    end
    if strcmp(algo, 'fit') && ~license('test', 'curve_fitting_toolbox')
        warning('f_gauss_fit:noCFToolbox', ...
            'Curve Fitting Toolbox 不可用，自动降级为 fminsearch。');
        algo = 'fminsearch';
    end

    %% ── 执行拟合 ──────────────────────────────────────────────────────────
    converged = false;
    p_norm    = x0_norm;

    switch algo

        %-- lsqcurvefit ───────────────────────────────────────────────────
        case 'lsqcurvefit'
            lsq_opts = optimoptions('lsqcurvefit', ...
                'Display',                'off', ...
                'FunctionTolerance',      1e-12, ...
                'StepTolerance',          1e-12, ...
                'MaxFunctionEvaluations', 5000);
            try
                [p_norm, ~, ~, exitflag] = lsqcurvefit( ...
                    gauss_fn, x0_norm, t_norm, y_norm, ...
                    lb_norm, ub_norm, lsq_opts);
                converged = exitflag > 0;
            catch ME
                warning('f_gauss_fit:lsqFailed', ...
                    'lsqcurvefit 执行失败（%s），降级为 fminsearch。', ME.message);
                algo = 'fminsearch';
            end

        %-- fit（Curve Fitting Toolbox）──────────────────────────────────
        case 'fit'
            try
                if use_baseline
                    ft = fittype('A*exp(-(x-mu)^2/(2*s^2))+C', ...
                        'independent', 'x', ...
                        'coefficients', {'A','mu','s','C'});
                    fo = fitoptions(ft, ...
                        'StartPoint', x0_norm,   ...
                        'Lower',      lb_norm,   ...
                        'Upper',      ub_norm,   ...
                        'Display',    'off',     ...
                        'MaxIter',    1000);
                    fobj   = fit(t_norm(:), y_norm(:), ft, fo);
                    p_norm = [fobj.A, fobj.mu, fobj.s, fobj.C];
                else
                    ft = fittype('A*exp(-(x-mu)^2/(2*s^2))', ...
                        'independent', 'x', ...
                        'coefficients', {'A','mu','s'});
                    fo = fitoptions(ft, ...
                        'StartPoint', x0_norm,      ...
                        'Lower',      lb_norm(1:3), ...
                        'Upper',      ub_norm(1:3), ...
                        'Display',    'off',        ...
                        'MaxIter',    1000);
                    fobj   = fit(t_norm(:), y_norm(:), ft, fo);
                    p_norm = [fobj.A, fobj.mu, fobj.s];
                end
                converged = true;
            catch ME
                warning('f_gauss_fit:fitFailed', ...
                    'fit 执行失败（%s），降级为 fminsearch。', ME.message);
                algo = 'fminsearch';
            end

    end   % switch

    %-- fminsearch（兜底，或从上方降级流入）──────────────────────────────
    if strcmp(algo, 'fminsearch')
        fm_opts = optimset( ...
            'Display',     'off',  ...
            'MaxFunEvals', 10000,  ...
            'MaxIter',     5000,   ...
            'TolFun',      1e-12,  ...
            'TolX',        1e-12);
        % fminsearch 不支持显式约束；对 sigma 取绝对值保证正定性
        cost_unc = @(p) cost_fn([p(1), p(2), abs(p(3)), p(4:end)]);
        [p_raw, ~, exitflag] = fminsearch(cost_unc, x0_norm, fm_opts);
        p_norm    = p_raw;
        p_norm(3) = abs(p_norm(3));
        converged = (exitflag == 1);
    end

    %% ── 反归一化：还原至物理量 ────────────────────────────────────────────
    A_fit     =  p_norm(1) * y_scale;
    mu_fit    =  p_norm(2) * t_scale + t_offset;
    sigma_fit =  abs(p_norm(3)) * t_scale;
    if use_baseline
        C_fit = p_norm(4) * y_scale;
    end

    %% ── 拟合质量指标（在原始物理域计算）─────────────────────────────────
    if use_baseline
        y_hat = A_fit .* exp(-(t_raw - mu_fit).^2 ./ (2 .* sigma_fit.^2)) + C_fit;
    else
        y_hat = A_fit .* exp(-(t_raw - mu_fit).^2 ./ (2 .* sigma_fit.^2));
    end

    residuals = y_raw - y_hat;
    rmse      = sqrt(mean(residuals.^2));
    ss_res    = sum(residuals.^2);
    ss_tot    = sum((y_raw - mean(y_raw)).^2);
    if ss_tot < eps
        r_squared = 0;
    else
        r_squared = 1 - ss_res / ss_tot;
    end

    %% ── 组装输出结构体 ────────────────────────────────────────────────────
    result.A         = A_fit;
    result.mu        = mu_fit;
    result.sigma     = sigma_fit;
    if use_baseline
        result.C     = C_fit;
    end
    result.fwhm      = 2 * sqrt(2 * log(2)) * sigma_fit;
    result.residual  = rmse;
    result.r_squared = r_squared;
    result.algorithm = algo;
    result.converged = converged;
    result.x_fit     = t_raw;
    result.y_fit     = y_hat;

    %% ── 可选绘图 ──────────────────────────────────────────────────────────
    if options.Plot
        t_ms = t_raw * 1e3;

        figure('Name', '高斯拟合结果', 'NumberTitle', 'off');

        subplot(2, 1, 1);
        plot(t_ms, y_raw, 'Color', [0.6 0.6 0.6], ...
             'LineWidth', 0.8, 'DisplayName', '原始信号');
        hold on;
        plot(t_ms, y_hat, 'r-', 'LineWidth', 2, 'DisplayName', '高斯拟合');
        xline(mu_fit * 1e3, 'b--', ...
              sprintf('\\mu = %.4g ms', mu_fit * 1e3), ...
              'LabelVerticalAlignment', 'bottom', 'LineWidth', 1);
        xlabel('时间 (ms)');
        ylabel('幅值 (V)');
        legend('Location', 'best');
        title(sprintf('高斯拟合  |  算法: %s  |  R² = %.6f', algo, r_squared));
        grid on;  box on;

        subplot(2, 1, 2);
        plot(t_ms, residuals, 'Color', [0.15 0.55 0.25], 'LineWidth', 0.8);
        yline(0, 'k--');
        xlabel('时间 (ms)');
        ylabel('残差 (V)');
        title(sprintf('拟合残差  |  RMSE = %.4g V', rmse));
        grid on;  box on;

        sgtitle(sprintf( ...
            'A = %.4g V  |  \\mu = %.4g ms  |  \\sigma = %.4g \\mus  |  FWHM = %.4g \\mus', ...
            A_fit, mu_fit * 1e3, sigma_fit * 1e6, result.fwhm * 1e6), ...
            'FontSize', 11);
    end

end
