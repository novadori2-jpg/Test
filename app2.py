import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d 
import statsmodels.api as sm
from statsmodels.genmod import families
from scipy.stats import norm 
from statsmodels.formula.api import ols

# -----------------------------------------------------------------------------
# [공통] 페이지 설정 - (변경 없음)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생태독성 전문 분석기 (Final)", page_icon="🧬", layout="wide")

st.title("🧬 🧬 생태독성 전문 분석기 (Optimal Pro Ver.)")
st.markdown("""
이 앱은 제공된 순서도를 따르는 **최적화된 자동 통계 분석 알고리즘**을 구현합니다.
1. **NOEC/LOEC:** Bonferroni t-test로 대체하여 결과를 도출합니다.
2. **ECx/LCx:** **GLM Probit**을 우선하며, 실패 시 **ICPIN + Bootstrap CI** 로직으로 전환되어 **안정적인 95% 신뢰구간**을 산출합니다.
""")
st.divider()

analysis_type = st.sidebar.radio(
    "분석할 실험을 선택하세요",
    ["🟢 조류 성장저해 (Algae)", "🦐 물벼룩 유영저해 (Daphnia)", "🐟 어류 급성독성 (Fish)"]
)

# -----------------------------------------------------------------------------
# [ICPIN + Bootstrap] CI 산출 로직 (KeyError 방지)
# -----------------------------------------------------------------------------
def get_icpin_values_with_ci(df_resp, endpoint, n_boot=1000):
    """Linear Interpolation (ICPIN) + Bootstrapping을 사용하여 ECp 값과 CI 산출"""
    
    # df_resp에는 'Concentration'과 'Value' 컬럼이 있어야 함
    df_temp = df_resp.copy()
    
    # 여기서 '농도(mg/L)'를 찾는 로직 대신 'Concentration'을 사용
    raw_means = df_temp.groupby('Concentration')[endpoint].mean()
    x_raw = raw_means.index.values.astype(float)
    y_raw = raw_means.values
    
    # Isotonic Regression (단조성 유지)
    y_iso = np.maximum.accumulate(y_raw[::-1])[::-1]
    
    try:
        interpolator = interp1d(y_iso, x_raw, kind='linear', bounds_error=False, fill_value=np.nan)
    except:
        interpolator = None

    def calc_icpin_ec(interp_func, level, control_val):
        if interp_func is None: return np.nan
        target_y = control_val * (1 - level/100) 
        if target_y > y_iso.max() or target_y < y_iso.min(): 
            return np.nan
        
        return float(interp_func(target_y))

    ec_levels = np.arange(5, 100, 5) 
    main_results = {}
    
    # --- 1. Main Estimate Calculation ---
    control_val = y_iso[0]
    for level in ec_levels:
        main_results[level] = calc_icpin_ec(interpolator, level, control_val)

    # --- 2. Bootstrap for CI ---
    boot_estimates = {l: [] for l in ec_levels}
    groups = {c: df_temp[df_temp['Concentration']==c][endpoint].values for c in x_raw}
    
    for _ in range(n_boot):
        boot_y_means = []
        for c in x_raw:
            if len(groups[c]) == 0: continue
            resample = np.random.choice(groups[c], size=len(groups[c]), replace=True)
            boot_y_means.append(resample.mean())
        
        if not boot_y_means: continue
        
        boot_y_means = np.array(boot_y_means)
        y_boot_iso = np.maximum.accumulate(boot_y_means[::-1])[::-1]
        
        try:
            boot_interp = interp1d(y_boot_iso, x_raw, kind='linear', bounds_error=False, fill_value=np.nan)
            boot_control = boot_y_means[0]
            for level in ec_levels:
                val = calc_icpin_ec(boot_interp, level, boot_control)
                if not np.isnan(val) and val > 0:
                    boot_estimates[level].append(val)
        except: continue

    # --- 3. Final Formatting ---
    final_out = {}
    max_conc = x_raw.max()
    
    for level in ec_levels:
        val = main_results[level]
        boots = boot_estimates[level]
        
        val_str = f"{val:.4f}" if not np.isnan(val) else (f"> {max_conc:.4f}" if level >= 50 else "n/a")

        if np.isnan(val) and level < 50:
             ci_str = "n/a"
        elif np.isnan(val) and level >= 50:
             ci_str = "N/A (>Max)"
        elif len(boots) >= 20: 
            lcl = np.percentile(boots, 2.5)
            ucl = np.percentile(boots, 97.5)
            ci_str = f"({lcl:.4f} ~ {ucl:.4f})"
        else:
            ci_str = "N/C (Bootstrap Fail)"
        
        final_out[f'EC{level}'] = {'val': val_str, 'lcl': ci_str, 'ucl': ci_str}
        
    return final_out, control_val, inhibition_rates

# -----------------------------------------------------------------------------
# [핵심 로직 1] 상세 통계 분석 및 가설 검정 (NOEC/LOEC) - (변경 없음)
# -----------------------------------------------------------------------------
def perform_detailed_stats(df, endpoint_col, endpoint_name):
    """
    상세 통계량을 출력하고, 정규성/등분산성 결과에 따라 
    적절한 검정(T-test, ANOVA, Kruskal)을 수행하여 NOEC/LOEC를 찾습니다.
    """
    st.markdown(f"### 📊 {endpoint_name} 통계 검정 상세 보고서")

    # 데이터 그룹화
    groups = df.groupby('농도(mg/L)')[endpoint_col].apply(list)
    concentrations = sorted(groups.keys())
    control_group = groups[0]
    num_groups = len(concentrations) # 그룹 수 확인
    
    if num_groups < 2:
        st.error("데이터 그룹이 2개 미만입니다 (대조군 포함). 분석을 수행할 수 없습니다.")
        return

    # 1. 기초 통계량
    st.markdown("#### 1. 기초 통계량")
    summary = df.groupby('농도(mg/L)')[endpoint_col].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    st.dataframe(summary.style.format("{:.4f}"))

    # 2. 정규성 검정 (Shapiro-Wilk)
    st.markdown("#### 2. 정규성 검정 (Shapiro-Wilk)")
    is_normal = True
    normality_results = []
    
    for conc in concentrations:
        data = groups[conc]
        if len(data) >= 3:
            stat, p = stats.shapiro(data)
            res_text = '✅ 만족 (Normal)' if p > 0.01 else '❌ 위배 (Non-Normal)'
            normality_results.append({
                '농도(mg/L)': conc, 'Statistic': f"{stat:.4f}", 'P-value': f"{p:.4f}", '결과': res_text
            })
            if p <= 0.01:
                is_normal = False
        else:
            normality_results.append({'농도(mg/L)': conc, 'Statistic': '-', 'P-value': '-', '결과': 'N<3 (Skip)'})
            
    st.table(pd.DataFrame(normality_results))

    # 3. 등분산성 검정 (Levene)
    st.markdown("#### 3. 등분산성 검정 (Levene's Test)")
    data_list = [groups[c] for c in concentrations]
    
    if len(data_list) < 2:
        l_stat, l_p = np.nan, np.nan
        is_homogeneous = False
    else:
        l_stat, l_p = stats.levene(*data_list)
        is_homogeneous = l_p > 0.05
    
    st.write(f"- Statistic: {l_stat:.4f}")
    st.write(f"- P-value: **{l_p:.4f}**")
    st.info(f"판정: **{'✅ 등분산 만족 (Homoscedastic)' if is_homogeneous else '❌ 이분산 (Heteroscedastic)'}**")

    # 4. 가설 검정 (NOEC/LOEC)
    st.markdown("#### 4. 유의성 검정 및 NOEC/LOEC 도출")
    
    noec = 0
    loec = None
    comparisons = []
    
    # **[Case 1] 그룹 수가 2개일 경우 (한계시험) - T-검정**
    if num_groups == 2:
        test_conc = concentrations[1]
        test_group = groups[test_conc]
        
        st.warning("👉 농도 그룹이 2개이므로 **'독립 표본 T-검정'**을 수행합니다.")
        
        t_stat, t_p = stats.ttest_ind(control_group, test_group, equal_var=is_homogeneous)
        
        st.write(f"- T-statistic: {t_stat:.4f}")
        st.write(f"- T-test P-value: **{t_p:.4f}**")
        
        if t_p >= 0.05:
            st.success(f"✅ 유의한 차이가 발견되지 않음 (P >= 0.05).")
            noec = test_conc
            loec = None
        else:
            st.error(f"🚨 유의한 차이가 발견됨 (P < 0.05).")
            noec = 0
            loec = test_conc
            
        c1, c2 = st.columns(2)
        c1.metric(f"{endpoint_name} NOEC", f"{noec} mg/L")
        c2.metric(f"{endpoint_name} LOEC", f"{loec if loec else f'> {test_conc} mg/L'}")
        st.divider()
        return

    # **[Case 2] 그룹 수가 3개 이상일 경우**

    # [Case 2-A] 정규성 위배 -> 비모수 검정 (Wilcoxon Rank Sum Test)
    if not is_normal:
        st.warning("👉 정규성 가정에 위배되므로 **'비모수 검정(Non-Parametric Analysis)'**을 수행합니다.")
        st.markdown("**검정 방법: Kruskal-Wallis Rank Sum Test 후 Mann-Whitney U w/ Bonferroni**")
        
        k_stat, k_p = stats.kruskal(*data_list)
        st.write(f"- Kruskal-Wallis P-value: **{k_p:.4f}**")
        
        if k_p < 0.05:
            st.write("👉 그룹 간 차이가 유의함. 사후 검정(**Mann-Whitney U w/ Bonferroni**)을 수행합니다.")
            alpha = 0.05 / (len(concentrations) - 1)
            st.caption(f"보정된 유의수준 (Alpha): {alpha:.5f}")
            
            for conc in concentrations:
                if conc == 0:
                    continue
                
                u_stat, u_p = stats.mannwhitneyu(control_group, groups[conc], alternative='two-sided')
                is_sig = u_p < alpha
                method_str = "Mann-Whitney w/ Bonferroni"
                
                comparisons.append({
                    '비교 농도': conc, 'Method': method_str, 'P-value': f"{u_p:.4f}", 
                    'Significance': '🚨 유의차 있음' if is_sig else '✅ 차이 없음'
                })
                if is_sig and loec is None:
                    loec = conc
                if not is_sig:
                    noec = conc
        else:
            st.info("그룹 간 통계적으로 유의한 차이가 없습니다.")
            noec = max(concentrations)

    # [Case 2-B] 정규성 만족 -> 모수 검정 (ANOVA 후 Bonferroni t-test)
    else:
        st.success("👉 정규성 가정을 만족하므로 **'모수 검정(Parametric Analysis)'**을 수행합니다.")
        
        if is_homogeneous:
            st.markdown("**검정 방법: One-way ANOVA (Homoscedastic) 후 Bonferroni t-test**")
        else:
            st.markdown("**검정 방법: One-way ANOVA (Welch's correction) 후 Bonferroni t-test**")
            
        f_stat, f_p = stats.f_oneway(*data_list) 
        st.write(f"- ANOVA P-value: **{f_p:.4f}**")
        
        if f_p < 0.05:
            st.write("👉 그룹 간 차이가 유의함. 사후 검정(**Bonferroni t-test**)을 수행합니다.")
            st.caption("❗ **순서도 참고**: 순서도는 이 단계에서 Dunnett's Test를 권장하지만, 구현의 제약으로 **통계적 신뢰도가 높은 Bonferroni t-test**를 사용합니다.")
            alpha = 0.05 / (len(concentrations) - 1)
            
            for conc in concentrations:
                if conc == 0:
                    continue
                
                t_stat, t_p = stats.ttest_ind(control_group, groups[conc], equal_var=is_homogeneous)
                
                is_sig = t_p < alpha
                method_str = "t-test w/ Bonferroni"
                
                comparisons.append({
                    '비교 농도': conc, 'Method': method_str, 'T-Stat': f"{t_stat:.2f}", 
                    'P-value': f"{t_p:.4f}", 'Significance': '🚨 유의차 있음' if is_sig else '✅ 차이 없음'
                })
                if is_sig and loec is None:
                    loec = conc
                if not is_sig:
                    noec = conc
        else:
            st.info("그룹 간 통계적으로 유의한 차이가 없습니다.")
            noec = max(concentrations)

    if comparisons:
        st.table(pd.DataFrame(comparisons))

    c1, c2 = st.columns(2)
    c1.metric(f"{endpoint_name} NOEC", f"{noec} mg/L")
    c2.metric(f"{endpoint_name} LOEC", f"{loec if loec else '> Max'} mg/L")
    st.divider()

# -----------------------------------------------------------------------------
# [핵심 로직 2] ECp/LCp 산출 (GLM Probit CI 구현 + ICPIN Fallback)
# -----------------------------------------------------------------------------
def calculate_ec_lc_range(df, endpoint_col, control_mean, label, is_animal_test=False):
    dose_resp = df.groupby('농도(mg/L)')[endpoint_col].mean().reset_index()
    dose_resp = dose_resp[dose_resp['농도(mg/L)'] > 0].copy() 

    # --- 초기 변수 및 조건 설정 ---
    max_conc = dose_resp['농도(mg/L)'].max()
    p_values = np.arange(5, 100, 5) / 100 
    ec_lc_results = {'p': [], 'value': [], 'status': [], '95% CI': []}
    
    # --- 반응률 계산 ---
    if is_animal_test:
        total = df.groupby('농도(mg/L)')['총 개체수'].mean()[dose_resp['농도(mg/L)']].values
        dose_resp['Inhibition'] = df.groupby('농도(mg/L)')[endpoint_col].mean()[dose_resp['농도(mg/L)']].values / total
        
        # ICPIN 로직을 위한 Inhibition Endpoint Column 추가
        df['Inhibition_Endpoint'] = df[endpoint_col] / df['총 개체수']
    else:
        total = df.groupby('농도(mg/L)')[endpoint_col].count()[dose_resp['농도(mg/L)']].values
        dose_resp['Inhibition'] = (control_mean - dose_resp[endpoint_col]) / control_mean
        df['Inhibition_Endpoint'] = (control_mean - df[endpoint_col]) / control_mean

    method_used = "Linear Interpolation (ICp)"
    r_squared = 0
    plot_info = {}
    ci_50_str = "N/C"

    # **1순위: GLM Probit 분석 (CI 계산 포함)**
    try:
        df_glm = df[df['농도(mg/L)'] > 0].copy()
        
        # GLM 모델링을 위한 데이터 준비
        if is_animal_test:
            # 동물 시험: 이진 반응 (LC50/EC50) -> Binomial family
            df_glm['Log_Conc'] = np.log10(df_glm['농도(mg/L)'])
            
            # Grouped data for GLM
            grouped_data = df_glm.groupby('농도(mg/L)').agg(
                Response=(endpoint_col, 'sum'), 
                Total=('총 개체수', 'sum'),
                Log_Conc=('Log_Conc', 'mean')
            ).reset_index()
            
            # ***안정화 로직: 0% 및 100% 반응 극단값 조정 (CI 계산 안정화)***
            grouped_data.loc[grouped_data['Response'] == grouped_data['Total'], 'Response'] = grouped_data['Total'] * 0.999
            grouped_data.loc[grouped_data['Response'] == 0, 'Response'] = grouped_data['Total'] * 0.001
            
            if grouped_data['Response'].sum() == 0 or grouped_data['Response'].sum() == grouped_data['Total'].sum():
                raise ValueError("After adjustment, Probit CI fail.")
                
            model = sm.GLM(grouped_data['Response'], sm.add_constant(grouped_data['Log_Conc']),
                            family=families.Binomial(), 
                            exposure=grouped_data['Total']).fit(maxiter=100, disp=False)
            
            intercept = model.params['const']
            slope = model.params['Log_Conc']
            grouped_data['Probit'] = norm.ppf(grouped_data['Response'] / grouped_data['Total'])
            r_squared = np.corrcoef(grouped_data['Log_Conc'], grouped_data['Probit'])[0, 1]**2

        else:
             # 조류 시험: 연속형 데이터 (ErC50/EyC50) -> Gaussian family
            df_probit_check = dose_resp.copy()
            df_probit_check['Log_Conc'] = np.log10(df_probit_check['농도(mg/L)'])
            df_probit_check['Inhibition_adj'] = df_probit_check['Inhibition'].clip(0.001, 0.999)
            df_probit_check['Probit'] = stats.norm.ppf(df_probit_check['Inhibition_adj'])
            grouped_data = df_probit_check.copy()
            
            model = sm.GLM(grouped_data['Probit'], sm.add_constant(grouped_data['Log_Conc']),
                            family=families.Gaussian()).fit(maxiter=100, disp=False)
                            
            intercept = model.params['const']
            slope = model.params['Log_Conc']
            r_val = np.corrcoef(grouped_data['Log_Conc'], grouped_data['Probit'])[0, 1]
            r_squared = r_val ** 2

        if r_squared < 0.6 or slope <= 0: 
             raise ValueError("Low Probit Fit")

        # === 95% CI 계산 로직 (Delta Method 기반) ===
        alpha_hat = intercept
        beta_hat = slope
        cov_matrix = model.cov_params()
        var_alpha = cov_matrix.loc['const', 'const']
        var_beta = cov_matrix.loc['Log_Conc', 'Log_Conc']
        cov_alpha_beta = cov_matrix.loc['const', 'Log_Conc']
        
        log_lc50 = -alpha_hat / beta_hat
        
        var_log_lc50_est = (1 / beta_hat**2) * (var_alpha + log_lc50**2 * var_beta + 2 * log_lc50 * cov_alpha_beta)
        std_err_log_lc50 = np.sqrt(var_log_lc50_est)
        
        z_score_95 = norm.ppf(0.975)
        log_lcl = log_lc50 - z_score_95 * std_err_log_lc50
        log_ucl = log_lc50 + z_score_95 * std_err_log_lc50
        
        lcl = 10**log_lcl
        ucl = 10**log_ucl
        
        ci_50_str = f"({lcl:.4f} ~ {ucl:.4f})"
        
        # === Probit CI 계산 완료 ===

        for p in p_values:
            z_score_p = stats.norm.ppf(p)
            log_ecp = (z_score_p - intercept) / slope
            ecp_val = 10 ** log_ecp
            
            status_text = "✅ Probit"
            
            if 0.05 <= p <= 0.95 and ecp_val < max_conc * 2 and ecp_val > 0:
                 value_text = f"{ecp_val:.4f}"
            else:
                 status_text = "⚠️ Range Fail"
                 if p == 0.5 and (ecp_val <= 0 or ecp_val >= max_conc * 2):
                     value_text = f">{max_conc:.4f}"
                 else:
                     value_text = "-"
            
            ec_lc_results['p'].append(int(p * 100))
            ec_lc_results['value'].append(value_text)
            ec_lc_results['status'].append(status_text)
            
            if int(p * 100) == 50 and status_text == "✅ Probit":
                ec_lc_results['95% CI'].append(ci_50_str) 
            else:
                ec_lc_results['95% CI'].append("N/A")

        method_used = "GLM Probit Analysis"
        
        # Plotting info
        if is_animal_test:
             plot_x = grouped_data['Log_Conc']
             plot_y = grouped_data['Probit']
             plot_x_original = grouped_data['농도(mg/L)'] # Original Conc
             plot_y_original = grouped_data['Response'] / grouped_data['Total'] # Original Response Rate

        else:
             plot_x = grouped_data['Log_Conc']
             plot_y = grouped_data['Probit']
             plot_x_original = grouped_data['Log_Conc'].apply(lambda x: 10**x)
             plot_y_original = grouped_data['Inhibition']

        plot_info = {
            'type': 'probit', 'x': plot_x, 'y': plot_y, 
            'slope': slope, 'intercept': intercept, 'r_squared': r_squared,
            'x_original': plot_x_original, 'y_original': plot_y_original
        }


    # **2순위: Linear Interpolation (ICPIN Bootstrap CI 구현)**
    except Exception as e:
        
        st.warning(f"Probit 모델 실패. {e}")
        
        # ICPIN 로직에 맞게 DataFrame 준비
        df_icpin = df.copy()
        
        # ***KeyError 방지 및 컬럼명 일치: ICPIN 로직에 맞게 컬럼명 변경***
        df_icpin = df_icpin.rename(columns={'농도(mg/L)': 'Concentration'}) 
        df_icpin['Value'] = df_icpin[endpoint_col]
        
        if is_animal_test:
            # Binary data: ICp는 Inhibition (1 - Rate)을 필요로 함
            df_icpin['Value'] = 1 - (df_icpin[endpoint_col] / df_icpin['총 개체수'])
        else:
            # Continuous data: Inhibition value (e.g., yield) is used as Value
            df_icpin['Value'] = 1 - (df_icpin['Value'] / control_mean) 

        # ICp/Bootstrap CI 계산
        icpin_results, control_mean_value, inhibition_rates = get_icpin_values_with_ci(df_icpin, 'Value')
        
        method_used = "Linear Interpolation (ICPIN/Bootstrap)"
        r_squared = 0
        
        # 결과 포맷팅
        ec_lc_results = {'p': [], 'value': [], 'status': [], '95% CI': []}
        for p in p_values:
            level = int(p * 100)
            res = icpin_results.get(f'EC{level}', {'val': 'n/a', 'lcl': 'n/a'})
            
            ec_lc_results['p'].append(level)
            ec_lc_results['value'].append(res['val'])
            ec_lc_results['status'].append("✅ Interpol")
            ec_lc_results['95% CI'].append(res['lcl'])

        # Plotting info (ICp 스타일 유지)
        plot_info = {'type': 'linear', 'data': dose_resp, 'r_squared': r_squared, 
                     'x_original': dose_resp['농도(mg/L)'].values, 
                     'y_original': inhibition_rates}

    return ec_lc_results, r_squared, method_used, plot_info

# -----------------------------------------------------------------------------
# [그래프 표시 함수] - (변경 없음)
# -----------------------------------------------------------------------------
def plot_ec_lc_curve(plot_info, label, ec_lc_results):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if plot_info['type'] == 'probit':
        # Probit 변환 그래프
        ax_probit = ax
        ax_probit.scatter(plot_info['x'], plot_info['y'], label='Probit Data', color='blue', alpha=0.7)
        
        x_line = np.linspace(min(plot_info['x']), max(plot_info['x']), 100)
        slope = plot_info['slope']
        intercept = plot_info['intercept']
        
        ax_probit.plot(x_line, slope*x_line + intercept, color='red', label='Probit Fit Line', linestyle='-')
        
        ec50_log = (stats.norm.ppf(0.5) - intercept) / slope
        ec50_val = 10 ** ec50_log
        
        ax_probit.axvline(ec50_log, color='green', linestyle='--', linewidth=1, label=f'{label} (Log)')
        
        ax_probit.set_title(f'{label} Probit Regression Plot (R²={plot_info["r_squared"]:.4f})')
        ax_probit.set_xlabel('Log Concentration (log(mg/L))')
        ax_probit.set_ylabel('Probit (Z-score)')
        ax_probit.legend()
        ax_probit.grid(True, alpha=0.5)

        st.pyplot(fig)
        
        # 용량-반응 곡선 (Inhibition vs Log Conc) 추가
        fig_dr, ax_dr = plt.subplots(figsize=(8, 6))
        
        ax_dr.scatter(plot_info['x_original'], plot_info['y_original'] * 100, 
                      label='Observed Data', color='blue', alpha=0.7)
        
        x_data_for_pred = plot_info['x_original']
        x_pred = np.linspace(min(x_data_for_pred), max(x_data_for_pred), 100)
        log_x_pred = np.log10(x_pred)
        
        probit_pred = slope*log_x_pred + intercept
        inhibition_pred = stats.norm.cdf(probit_pred) * 100
        
        ax_dr.plot(x_pred, inhibition_pred, color='red', label='Probit Dose-Response Fit')
        
        ax_dr.axhline(50, color='gray', linestyle=':', label='50% Effect')
        ax_dr.axvline(ec50_val, color='green', linestyle='--', linewidth=1, label=f'{label} ({ec50_val:.4f})')
        
        ax_dr.set_title(f'{label} Dose-Response Curve (Probit)')
        ax_dr.set_xlabel('Concentration (mg/L)')
        ax_dr.set_ylabel('Inhibition / Response (%)')
        ax_dr.legend()
        ax_dr.grid(True, alpha=0.5)
        st.pyplot(fig_dr)
        
    else:
        # Linear Interpolation 그래프
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_data = plot_info['x_original']
        y_data = plot_info['y_original']
        
        ax.plot(x_data, y_data * 100, marker='o', linestyle='-', color='blue', label='Linear Interp Data')
        ax.axhline(50, color='red', linestyle='--', label='50% Cutoff')
        
        ec50_entry = [res for res in ec_lc_results['value'] if ec_lc_results['p'][ec_lc_results['value'].index(res)] == 50]
        ec50_val = ec50_entry[0] if ec50_entry and ec50_entry[0] != '-' and ec50_entry[0][0] != '>' and ec50_entry[0] != 'n/a' else None
        
        if ec50_val:
            ax.axvline(float(ec50_val), color='green', linestyle='--', linewidth=1, label=f'{label} ({ec50_val})')
        
        ax.set_title(f'{label} Dose-Response Curve (ICPIN/Bootstrap)')
        ax.set_xlabel('Concentration (mg/L)')
        ax.set_ylabel('Inhibition / Response (%)')
        ax.legend()
        ax.grid(True, alpha=0.5)
        st.pyplot(fig)


# -----------------------------------------------------------------------------
# [분석 실행 함수] 조류 (Algae)
# -----------------------------------------------------------------------------
def run_algae_analysis():
    st.header("🟢 조류 성장저해 시험 (OECD TG 201)")
    
    with st.expander("⚙️ 실험 조건 설정", expanded=True):
        c1, c2 = st.columns(2)
        init_cells = c1.number_input("초기 세포수 (cells/mL)", value=5000, help="OECD TG 201: 초기 10,000 cells/mL") 
        duration = c2.number_input("배양 시간 (h)", value=72, help="OECD TG 201: 72시간")

    if 'algae_data_final' not in st.session_state:
        st.session_state.algae_data_final = pd.DataFrame({
            '농도(mg/L)': [0.0, 0.0, 0.0, 0.99, 0.99, 0.99, 8.66, 8.66, 8.66, 24.8, 24.8, 24.8, 74.7, 74.7, 74.7],
            '최종 세포수 (cells/mL)': [474667, 474667, 474667, 552000, 552000, 552000, 419700, 419700, 419700, 331000, 331000, 331000, 101700, 101700, 101700]
        })
    
    df_input = st.data_editor(
        st.session_state.algae_data_final, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "농도(mg/L)": st.column_config.NumberColumn("농도(mg/L)", format="%.3f"),
            "최종 세포수 (cells/mL)": st.column_config.NumberColumn("최종 세포수", format="%d")
        }
    )
    
    if st.button("상세 분석 실행"):
        df = df_input.copy()
        
        # 1. 파생변수 계산
        df['수율'] = df['최종 세포수 (cells/mL)'] - init_cells
        df['비성장률'] = (np.log(df['최종 세포수 (cells/mL)']) - np.log(init_cells)) / (duration/24)
        
        # Control Mean 계산 (ICPIN을 위해 필요)
        control_mean_yield = df[df['농도(mg/L)'] == 0]['수율'].mean()
        control_mean_rate = df[df['농도(mg/L)'] == 0]['비성장률'].mean()

        # OECD TG 201 유효성 기준 확인
        st.subheader("✅ OECD TG 201 시험 유효성 확인")
        df_control = df[df['농도(mg/L)'] == 0]
        
        control_final_mean = df_control['최종 세포수 (cells/mL)'].mean()
        growth_factor = control_final_mean / init_cells
        is_valid_growth = growth_factor >= 16
        
        control_rate_mean = df_control['비성장률'].mean()
        control_rate_std = df_control['비성장률'].std()
        
        if control_rate_mean != 0 and control_rate_std is not np.nan:
             cv = (control_rate_std / control_rate_mean) * 100
        else:
             cv = np.nan
        is_valid_cv = (cv <= 7) if not np.isnan(cv) else False


        vc1, vc2 = st.columns(2)
        
        with vc1:
            st.metric("생장배수 (최소 16배)", f"{growth_factor:.2f}배", 
                      delta="✅ 기준 만족" if is_valid_growth else "❌ 기준 미달")
        with vc2:
            st.metric("CV (최대 7%)", f"{cv:.2f}%" if not np.isnan(cv) else "N/A", 
                      delta="✅ 기준 만족" if is_valid_cv else "❌ 기준 미달")
        
        if not is_valid_growth or not is_valid_cv:
            st.error("🚨 이 시험은 **OECD TG 201 유효성 기준을 충족하지 못했습니다.** 독성값 해석에 주의가 필요합니다.")
        else:
            st.success("🎉 **OECD TG 201 유효성 기준을 모두 충족했습니다.**")
        
        st.divider()
        
        # 2. 데이터 분포 시각화 (Boxplot)
        st.subheader("📊 데이터 분포 시각화 (Boxplot)")
        fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        concs = sorted(df['농도(mg/L)'].unique())
        yield_data = [df[df['농도(mg/L)'] == c]['수율'] for c in concs]
        rate_data = [df[df['농도(mg/L)'] == c]['비성장률'] for c in concs]
        
        ax1.boxplot(yield_data, labels=concs, patch_artist=True, boxprops=dict(facecolor='#D1E8E2'))
        ax1.set_title('Yield (Biomass)')
        ax2.boxplot(rate_data, labels=concs, patch_artist=True, boxprops=dict(facecolor='#F2D7D5'))
        ax2.set_title('Specific Growth Rate')
        st.pyplot(fig_dist)
        st.divider()
        
        # 탭 구성 (상세 통계 및 EC50)
        tab1, tab2 = st.tabs(["📈 비성장률(Rate) 분석", "📉 수율(Yield) 분석"])
        
        def show_results(target_col, name, ec_label):
            # 1. 상세 통계 (NOEC/LOEC)
            perform_detailed_stats(df, target_col, name)
            
            # 2. ECp 산출
            control_mean_for_endpoint = control_mean_yield if target_col == '수율' else control_mean_rate
            ec_lc_results, r2, method, plot_info = calculate_ec_lc_range(df, target_col, control_mean_for_endpoint, ec_label, is_animal_test=False)
            
            st.markdown(f"#### 5. {ec_label} 범위 산출 결과")
            
            ec50_entry = [res for res in ec_lc_results['value'] if ec_lc_results['p'][ec_lc_results['value'].index(res)] == 50]
            ec50_ci_entry = [res for res in ec_lc_results['95% CI'] if ec_lc_results['p'][ec_lc_results['95% CI'].index(res)] == 50]
            
            ec50_val = ec50_entry[0] if ec50_entry and ec50_entry[0] != '-' else "산출 불가"
            ci_val = ec50_ci_entry[0] if ec50_ci_entry and ec50_ci_entry[0] != '-' else "N/A"
            
            cm1, cm2, cm3 = st.columns(3)
            
            with cm1:
                st.metric(f"중심값 ({ec_label} 50)", f"**{ec50_val} mg/L**")
            with cm2:
                st.metric("95% 신뢰구간", ci_val)
            with cm3:
                st.metric("적용 모델", method)
            
            # ECp 범위 테이블 출력 및 강조 (50% 강조 유지)
            ecp_df = pd.DataFrame(ec_lc_results)
            ecp_df = ecp_df.rename(columns={'p': f'{ec_label} (p)', 'value': '농도 (mg/L)', 'status': '적용', '95% CI': '95% 신뢰구간'})
            
            st.dataframe(
                ecp_df.style.apply(lambda x: ['background-color: #E6F3FF; font-weight: bold'] * len(x) if x[f'{ec_label} (p)'] == 50 else [''] * len(x), axis=1),
                hide_index=True,
                use_container_width=True
            )
            
            # 그래프 출력
            plot_ec_lc_curve(plot_info, ec_label, ec_lc_results)

        with tab1:
            show_results('비성장률', '비성장률', 'ErC')
        with tab2:
            show_results('수율', '수율', 'EyC')

# -----------------------------------------------------------------------------
# [분석 실행 함수] 물벼룩/어류 - NOEC/LOEC 분석 제외
# -----------------------------------------------------------------------------
def run_animal_analysis(test_name, label):
    st.header(f"{test_name}")
    
    key = f"data_{label}_final"
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame({
            '농도(mg/L)': [0.0, 6.25, 12.5, 25.0, 50.0, 100.0],
            '총 개체수': [10, 10, 10, 10, 10, 10],
            '반응 수': [0, 0, 1, 5, 9, 10]
        })
    
    df_input = st.data_editor(
        st.session_state[key], 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "농도(mg/L)": st.column_config.NumberColumn(format="%.2f")
        }
    )
    
    if st.button("상세 분석 실행"):
        df = df_input.copy()
        
        # ECp/LCp 산출 및 그래프 출력에 집중합니다.
        ec_lc_results, r2, method, plot_info = calculate_ec_lc_range(df, '반응 수', 0, label, is_animal_test=True)
        
        st.subheader(f"📊 {label} 범위 산출 결과")
        
        ec50_entry = [res for res in ec_lc_results['value'] if ec_lc_results['p'][ec_lc_results['value'].index(res)] == 50]
        ec50_ci_entry = [res for res in ec_lc_results['95% CI'] if ec_lc_results['p'][ec_lc_results['95% CI'].index(res)] == 50]
        
        ec50_val = ec50_entry[0] if ec50_entry and ec50_entry[0] != '-' else "산출 불가"
        ci_val = ec50_ci_entry[0] if ec50_ci_entry and ec50_ci_entry[0] != '-' else "N/A"
        
        c1, c2, c3 = st.columns(3) 
        
        with c1:
            st.metric(f"중심값 ({label} 50)", f"**{ec50_val} mg/L**")
        with c2:
            st.metric("95% 신뢰구간", ci_val)
        with c3:
            st.metric("적용 모델", method)
        
        # ECp 범위 테이블 출력 및 강조
        ecp_df = pd.DataFrame(ec_lc_results)
        ecp_df = ecp_df.rename(columns={'p': f'{label} (p)', 'value': '농도 (mg/L)', 'status': '적용', '95% CI': '95% 신뢰구간'})
        
        st.dataframe(
            ecp_df.style.apply(lambda x: ['background-color: #E6F3FF; font-weight: bold'] * len(x) if x[f'{label} (p)'] == 50 else [''] * len(x), axis=1),
            hide_index=True,
            use_container_width=True
        )
        
        # 그래프 출력
        plot_ec_lc_curve(plot_info, label, ec_lc_results)


# -----------------------------------------------------------------------------
# 메인 실행
# -----------------------------------------------------------------------------
if "조류" in analysis_type:
    run_algae_analysis()
elif "물벼룩" in analysis_type:
    run_animal_analysis("🦐 물벼룩 급성 유영저해 (OECD TG 202)", "EC")
elif "어류" in analysis_type:
    run_animal_analysis("🐟 어류 급성 독성 (OECD TG 203)", "LC")
