import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.genmod import families
from scipy.stats import norm 
from scipy.interpolate import interp1d 
from statsmodels.formula.api import ols

# -----------------------------------------------------------------------------
# [공통] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생태독성 전문 분석기 (Final)", page_icon="🧬", layout="wide")

st.title("🧬 생태독성 전문 분석기 (Optimal Pro Ver.)")
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
# [ICPIN + Bootstrap] CI 산출 로직
# -----------------------------------------------------------------------------
def get_icpin_values_with_ci(df_resp, endpoint, n_boot=1000):
    """Linear Interpolation (ICPIN) + Bootstrapping을 사용하여 ECp 값과 CI 산출"""
    
    df_temp = df_resp.copy()
    
    raw_means = df_temp.groupby('Concentration')[endpoint].mean()
    x_raw = raw_means.index.values.astype(float)
    y_raw = raw_means.values
    
    # Isotonic Regression
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
    
    # *** Inhibition Rates 계산 (그래프용) ***
    # control_val이 0일 경우 0으로 나누는 에러 방지
    if control_val == 0:
        inhibition_rates = np.zeros_like(y_raw)
    else:
        inhibition_rates = (control_val - y_raw) / control_val
    
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
# [핵심 로직 1] 상세 통계 분석 및 가설 검정 (NOEC/LOEC)
# -----------------------------------------------------------------------------
def perform_detailed_stats(df, endpoint_col, endpoint_name):
    st.markdown(f"### 📊 {endpoint_name} 통계 검정 상세 보고서")

    groups = df.groupby('농도(mg/L)')[endpoint_col].apply(list)
    concentrations = sorted(groups.keys())
    control_group = groups[0]
    num_groups = len(concentrations)
    
    if num_groups < 2:
        st.error("데이터 그룹이 2개 미만입니다 (대조군 포함). 분석을 수행할 수 없습니다.")
        return

    st.markdown("#### 1. 기초 통계량")
    summary = df.groupby('농도(mg/L)')[endpoint_col].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    st.dataframe(summary.style.format("{:.4f}"))

    st.markdown("#### 2. 정규성 검정 (Shapiro-Wilk)")
    is_normal = True
    normality_results = []
    
    for conc in concentrations:
        data = groups[conc]
        if len(data) >= 3:
            stat, p = stats.shapiro(data)
            res_text = '✅ 만족 (Normal)' if p > 0.01 else '❌ 위배 (Non-Normal)'
            normality_results.append({'농도(mg/L)': conc, 'Statistic': f"{stat:.4f}", 'P-value': f"{p:.4f}", '결과': res_text})
            if p <= 0.01: is_normal = False
        else:
            normality_results.append({'농도(mg/L)': conc, 'Statistic': '-', 'P-value': '-', '결과': 'N<3 (Skip)'})
    st.table(pd.DataFrame(normality_results))

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

    st.markdown("#### 4. 유의성 검정 및 NOEC/LOEC 도출")
    noec = 0
    loec = None
    comparisons = []
    
    if num_groups == 2:
        test_conc = concentrations[1]
        test_group = groups[test_conc]
        st.warning("👉 농도 그룹이 2개이므로 **'독립 표본 T-검정'**을 수행합니다.")
        t_stat, t_p = stats.ttest_ind(control_group, test_group, equal_var=is_homogeneous)
        st.write(f"- T-test P-value: **{t_p:.4f}**")
        
        if t_p >= 0.05:
            st.success(f"✅ 유의한 차이가 발견되지 않음 (P >= 0.05).")
            noec = test_conc
        else:
            st.error(f"🚨 유의한 차이가 발견됨 (P < 0.05).")
            noec = 0
            loec = test_conc
        c1, c2 = st.columns(2)
        c1.metric(f"{endpoint_name} NOEC", f"{noec} mg/L")
        c2.metric(f"{endpoint_name} LOEC", f"{loec if loec else f'> {test_conc} mg/L'}")
        st.divider()
        return

    if not is_normal:
        st.warning("👉 정규성 가정 위배: **'비모수 검정(Kruskal-Wallis + Mann-Whitney)'**")
        k_stat, k_p = stats.kruskal(*data_list)
        st.write(f"- Kruskal-Wallis P-value: **{k_p:.4f}**")
        if k_p < 0.05:
            alpha = 0.05 / (len(concentrations) - 1)
            for conc in concentrations:
                if conc == 0: continue
                u_stat, u_p = stats.mannwhitneyu(control_group, groups[conc], alternative='two-sided')
                is_sig = u_p < alpha
                comparisons.append({'비교 농도': conc, 'Method': 'Mann-Whitney', 'P-value': f"{u_p:.4f}", 'Significance': '🚨 유의차 있음' if is_sig else '✅ 차이 없음'})
                if is_sig and loec is None: loec = conc
                if not is_sig: noec = conc
        else:
            noec = max(concentrations)
    else:
        st.success("👉 정규성 가정 만족: **'모수 검정(ANOVA + Bonferroni t-test)'**")
        f_stat, f_p = stats.f_oneway(*data_list) 
        st.write(f"- ANOVA P-value: **{f_p:.4f}**")
        if f_p < 0.05:
            alpha = 0.05 / (len(concentrations) - 1)
            for conc in concentrations:
                if conc == 0: continue
                t_stat, t_p = stats.ttest_ind(control_group, groups[conc], equal_var=is_homogeneous)
                is_sig = t_p < alpha
                comparisons.append({'비교 농도': conc, 'Method': 't-test w/ Bonferroni', 'P-value': f"{t_p:.4f}", 'Significance': '🚨 유의차 있음' if is_sig else '✅ 차이 없음'})
                if is_sig and loec is None: loec = conc
                if not is_sig: noec = conc
        else:
            noec = max(concentrations)

    if comparisons: st.table(pd.DataFrame(comparisons))
    c1, c2 = st.columns(2)
    c1.metric(f"{endpoint_name} NOEC", f"{noec} mg/L")
    c2.metric(f"{endpoint_name} LOEC", f"{loec if loec else '> Max'} mg/L")
    st.divider()

# -----------------------------------------------------------------------------
# [핵심 로직 2] ECp/LCp 산출 (GLM Probit CI 구현 + ICPIN Fallback)
# -----------------------------------------------------------------------------
def calculate_ec_lc_range(df, endpoint_col, control_mean, label, is_animal_test=False):
    # 원본 데이터를 사용하여 Probit/ICPIN용 데이터 준비
    dose_resp = df.groupby('농도(mg/L)')[endpoint_col].mean().reset_index()
    
    # dose_resp에서 0농도 제거 (Probit용)
    dose_resp_probit = dose_resp[dose_resp['농도(mg/L)'] > 0].copy()
    
    max_conc = dose_resp['농도(mg/L)'].max()
    p_values = np.arange(5, 100, 5) / 100 
    ec_lc_results = {'p': [], 'value': [], 'status': [], '95% CI': []}
    
    # 반응률 계산
    if is_animal_test:
        total_mean = df.groupby('농도(mg/L)')['총 개체수'].mean()
        # dose_resp_probit용 (0 제외)
        total_probit = total_mean[dose_resp_probit['농도(mg/L)']].values
        dose_resp_probit['Inhibition'] = dose_resp_probit[endpoint_col] / total_probit
    else:
        dose_resp_probit['Inhibition'] = (control_mean - dose_resp_probit[endpoint_col]) / control_mean

    method_used = "Linear Interpolation (ICp)"
    r_squared = 0
    plot_info = {}
    ci_50_str = "N/C"

    # **1순위: GLM Probit 분석**
    try:
        df_glm = df[df['농도(mg/L)'] > 0].copy()
        
        if is_animal_test:
            df_glm['Log_Conc'] = np.log10(df_glm['농도(mg/L)'])
            grouped_data = df_glm.groupby('농도(mg/L)').agg(
                Response=(endpoint_col, 'sum'), 
                Total=('총 개체수', 'sum'),
                Log_Conc=('Log_Conc', 'mean')
            ).reset_index()
            
            # 0% / 100% 조정
            grouped_data.loc[grouped_data['Response'] == grouped_data['Total'], 'Response'] = grouped_data['Total'] * 0.999
            grouped_data.loc[grouped_data['Response'] == 0, 'Response'] = grouped_data['Total'] * 0.001
            
            if grouped_data['Response'].sum() == 0 or grouped_data['Response'].sum() == grouped_data['Total'].sum():
                raise ValueError("Probit CI fail.")
                
            model = sm.GLM(grouped_data['Response'], sm.add_constant(grouped_data['Log_Conc']),
                            family=families.Binomial(), exposure=grouped_data['Total']).fit(maxiter=100, disp=False)
            
            intercept = model.params['const']
            slope = model.params['Log_Conc']
            grouped_data['Probit'] = norm.ppf(grouped_data['Response'] / grouped_data['Total'])
            r_squared = np.corrcoef(grouped_data['Log_Conc'], grouped_data['Probit'])[0, 1]**2
        else:
            df_probit_check = dose_resp_probit.copy()
            df_probit_check['Log_Conc'] = np.log10(df_probit_check['농도(mg/L)'])
            df_probit_check['Inhibition_adj'] = df_probit_check['Inhibition'].clip(0.001, 0.999)
            df_probit_check['Probit'] = stats.norm.ppf(df_probit_check['Inhibition_adj'])
            
            model = sm.GLM(df_probit_check['Probit'], sm.add_constant(df_probit_check['Log_Conc']),
                            family=families.Gaussian()).fit(maxiter=100, disp=False)
            intercept = model.params['const']
            slope = model.params['Log_Conc']
            r_squared = np.corrcoef(df_probit_check['Log_Conc'], df_probit_check['Probit'])[0, 1]**2
            grouped_data = df_probit_check

        if r_squared < 0.6 or slope <= 0: raise ValueError("Low Probit Fit")

        # CI 계산 (Delta Method)
        cov = model.cov_params()
        log_lc50 = -intercept / slope
        var_log_lc50 = (1/slope**2) * (cov.loc['const','const'] + log_lc50**2*cov.loc['Log_Conc','Log_Conc'] + 2*log_lc50*cov.loc['const','Log_Conc'])
        se_log = np.sqrt(var_log_lc50)
        ci_50_str = f"({10**(log_lc50 - 1.96*se_log):.4f} ~ {10**(log_lc50 + 1.96*se_log):.4f})"

        for p in p_values:
            ecp = 10**((stats.norm.ppf(p) - intercept) / slope)
            val_str = f"{ecp:.4f}" if 0.05<=p<=0.95 and ecp<max_conc*2 and ecp>0 else "-"
            ec_lc_results['p'].append(int(p*100))
            ec_lc_results['value'].append(val_str)
            ec_lc_results['status'].append("✅ Probit")
            ec_lc_results['95% CI'].append(ci_50_str if int(p*100)==50 else "N/A")

        method_used = "GLM Probit Analysis"
        
        if is_animal_test:
             plot_x = grouped_data['Log_Conc']
             plot_y = grouped_data['Probit']
             plot_x_orig = grouped_data['농도(mg/L)']
             plot_y_orig = grouped_data['Response'] / grouped_data['Total']
        else:
             plot_x = grouped_data['Log_Conc']
             plot_y = grouped_data['Probit']
             plot_x_orig = grouped_data['농도(mg/L)']
             plot_y_orig = grouped_data['Inhibition']

        plot_info = {'type': 'probit', 'x': plot_x, 'y': plot_y, 'slope': slope, 'intercept': intercept, 
                     'r_squared': r_squared, 'x_original': plot_x_orig, 'y_original': plot_y_orig}

    # **2순위: Linear Interpolation (ICPIN/Bootstrap)**
    except Exception as e:
        st.warning(f"Probit 모델 실패 ({e}). ICPIN + Bootstrap CI로 전환합니다.")
        
        df_icpin = df.copy()
        df_icpin = df_icpin.rename(columns={'농도(mg/L)': 'Concentration'})
        df_icpin['Value'] = df_icpin[endpoint_col]
        
        if is_animal_test:
            df_icpin['Value'] = 1 - (df_icpin[endpoint_col] / df_icpin['총 개체수']) # Inhibition
        else:
            df_icpin['Value'] = 1 - (df_icpin['Value'] / control_mean) # Inhibition Yield

        icpin_results, control_val, inhibition_rates = get_icpin_values_with_ci(df_icpin, 'Value')
        
        method_used = "Linear Interpolation (ICPIN/Bootstrap)"
        ci_50_str = icpin_results['EC50']['lcl']
        ec50_val = icpin_results['EC50']['val']
        
        ec_lc_results = {'p': [], 'value': [], 'status': [], '95% CI': []}
        for p in p_values:
            level = int(p*100)
            res = icpin_results.get(f'EC{level}', {'val': 'n/a', 'lcl': 'n/a'})
            ec_lc_results['p'].append(level)
            ec_lc_results['value'].append(res['val'])
            ec_lc_results['status'].append("✅ Interpol")
            ec_lc_results['95% CI'].append(res['lcl'])
            
        # *** 중요 수정: x_original을 inhibition_rates의 길이와 맞춤 ***
        # inhibition_rates는 대조군을 포함한 모든 농도 그룹에 대해 계산됨
        all_concs = sorted(df['농도(mg/L)'].unique()) 
        
        plot_info = {'type': 'linear', 'data': dose_resp, 'r_squared': 0, 
                     'x_original': all_concs,  # 대조군 포함 모든 농도
                     'y_original': inhibition_rates} # 대조군 포함 모든 저해율

    return ec_lc_results, r_squared, method_used, plot_info

# -----------------------------------------------------------------------------
# [그래프 표시 함수]
# -----------------------------------------------------------------------------
def plot_ec_lc_curve(plot_info, label, ec_lc_results):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if plot_info['type'] == 'probit':
        ax.scatter(plot_info['x'], plot_info['y'], label='Probit Data', color='blue')
        x_line = np.linspace(min(plot_info['x']), max(plot_info['x']), 100)
        ax.plot(x_line, plot_info['slope']*x_line + plot_info['intercept'], color='red', label='Fit')
        
        ec50_log = (stats.norm.ppf(0.5) - plot_info['intercept']) / plot_info['slope']
        ec50_val = 10**ec50_log
        
        ax.axvline(ec50_log, color='green', linestyle='--', label=f'Log EC50')
        ax.set_xlabel('Log Concentration')
        ax.set_ylabel('Probit')
        
        st.pyplot(fig)
        
        # Dose-Response
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(plot_info['x_original'], plot_info['y_original']*100, label='Observed')
        x_pred = np.linspace(min(plot_info['x_original']), max(plot_info['x_original']), 100)
        y_pred = stats.norm.cdf(plot_info['slope']*np.log10(x_pred) + plot_info['intercept']) * 100
        ax2.plot(x_pred, y_pred, color='red', label='Fit')
        ax2.axhline(50, color='gray', linestyle=':')
        ax2.axvline(ec50_val, color='green', linestyle='--', label=f'EC50: {ec50_val:.4f}')
        ax2.set_xlabel('Concentration')
        ax2.set_ylabel('Response (%)')
        ax2.legend()
        st.pyplot(fig2)
        
    else:
        # Linear
        x_data = plot_info['x_original']
        y_data = plot_info['y_original']
        
        ax.plot(x_data, y_data * 100, marker='o', linestyle='-', color='blue', label='Data')
        ax.axhline(50, color='red', linestyle='--', label='50% Cutoff')
        
        # EC50 Value Extraction
        ec50_entry = [res for res in ec_lc_results['value'] if ec_lc_results['p'][ec_lc_results['value'].index(res)] == 50]
        ec50_val = ec50_entry[0] if ec50_entry and ec50_entry[0] != '-' and 'n/a' not in str(ec50_entry[0]).lower() and '>' not in str(ec50_entry[0]) else None
        
        if ec50_val:
             try:
                val_float = float(ec50_val)
                ax.axvline(val_float, color='green', linestyle='--', label=f'EC50: {val_float}')
             except: pass

        ax.set_title(f'{label} Dose-Response (ICPIN)')
        ax.set_xlabel('Concentration')
        ax.set_ylabel('Inhibition (%)')
        ax.legend()
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# [분석 실행 함수] - (기존 유지)
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
    
    df_input = st.data_editor(st.session_state[key], num_rows="dynamic", use_container_width=True)
    
    if st.button("상세 분석 실행"):
        df = df_input.copy()
        ec_lc_results, r2, method, plot_info = calculate_ec_lc_range(df, '반응 수', 0, label, is_animal_test=True)
        
        st.subheader(f"📊 {label} 범위 산출 결과")
        
        ec50_idx = [i for i, p in enumerate(ec_lc_results['p']) if p == 50][0]
        ec50_val = ec_lc_results['value'][ec50_idx]
        ci_val = ec_lc_results['95% CI'][ec50_idx]
        
        c1, c2, c3 = st.columns(3)
        with c1: st.metric(f"중심값 ({label} 50)", f"**{ec50_val} mg/L**")
        with c2: st.metric("95% 신뢰구간", ci_val)
        with c3: st.metric("적용 모델", method)
        
        ecp_df = pd.DataFrame(ec_lc_results).rename(columns={'p': f'{label} (p)', 'value': '농도', 'status': '적용', '95% CI': '95% 신뢰구간'})
        st.dataframe(ecp_df.style.apply(lambda x: ['background-color: #E6F3FF; font-weight: bold']*len(x) if x[f'{label} (p)']==50 else ['']*len(x), axis=1))
        
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
