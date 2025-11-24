import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.genmod import families
from scipy.stats import norm 
from scipy.interpolate import interp1d 

# -----------------------------------------------------------------------------
# [공통] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생태독성 전문 분석기 (Final)", page_icon="🧬", layout="wide")

st.title("🧬 생태독성 전문 분석기 (Optimal Pro Ver.)")
st.markdown("""
이 앱은 **OECD TG 201, 202, 203** 보고서 요구사항을 충족합니다.
1. **조류 (Algae):** 생장 곡선 및 72h ErC50/EyC50.
2. **물벼룩 (Daphnia):** 24h, 48h EC50.
3. **어류 (Fish):** 24h, 48h, 72h, 96h LC50 및 **96h 기울기(Slope)**, **농도-치사 곡선**.
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
    df_temp = df_resp.copy()
    
    if 'Concentration' not in df_temp.columns:
        conc_col = [c for c in df_temp.columns if '농도' in c or 'Conc' in c][0]
        df_temp = df_temp.rename(columns={conc_col: 'Concentration'})
        
    raw_means = df_temp.groupby('Concentration')[endpoint].mean()
    x_raw = raw_means.index.values.astype(float)
    y_raw = raw_means.values
    
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
    
    control_val = y_iso[0]
    for level in ec_levels:
        main_results[level] = calc_icpin_ec(interpolator, level, control_val)

    boot_estimates = {l: [] for l in ec_levels}
    groups = {}
    for c in x_raw:
        vals = df_temp[df_temp['Concentration']==c][endpoint].values
        groups[c] = vals
    
    for _ in range(n_boot):
        boot_y_means = []
        for c in x_raw:
            if len(groups[c]) == 0: 
                boot_y_means.append(0)
                continue
            resample = np.random.choice(groups[c], size=len(groups[c]), replace=True)
            boot_y_means.append(resample.mean())
        
        boot_y_means = np.array(boot_y_means)
        y_boot_iso = np.maximum.accumulate(boot_y_means[::-1])[::-1]
        
        try:
            boot_interp = interp1d(y_boot_iso, x_raw, kind='linear', bounds_error=False, fill_value=np.nan)
            boot_control = y_boot_iso[0]
            for level in ec_levels:
                val = calc_icpin_ec(boot_interp, level, boot_control)
                if not np.isnan(val) and val > 0:
                    boot_estimates[level].append(val)
        except: continue

    final_out = {}
    max_conc = x_raw.max()
    
    if control_val != 0:
        inhibition_rates = (control_val - y_raw) / control_val
    else:
        inhibition_rates = np.zeros_like(y_raw)
    
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
# [함수 2] 상세 통계 분석 (NOEC/LOEC)
# -----------------------------------------------------------------------------
def perform_detailed_stats(df, endpoint_col, endpoint_name):
    st.markdown(f"### 📊 {endpoint_name} 통계 검정 상세 보고서")

    groups = df.groupby('농도(mg/L)')[endpoint_col].apply(list)
    concentrations = sorted(groups.keys())
    control_group = groups[0]
    num_groups = len(concentrations)
    
    if num_groups < 2:
        st.error("데이터 그룹이 2개 미만입니다.")
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
            res = '✅ 만족' if p > 0.01 else '❌ 위배'
            normality_results.append({'농도': conc, 'P-value': f"{p:.4f}", '결과': res})
            if p <= 0.01: is_normal = False
        else:
            normality_results.append({'농도': conc, 'P-value': '-', '결과': 'N<3'})
    st.table(pd.DataFrame(normality_results))

    st.markdown("#### 3. 등분산성 검정 (Levene)")
    data_list = [groups[c] for c in concentrations]
    if len(data_list) < 2:
        is_homogeneous = False
    else:
        l_stat, l_p = stats.levene(*data_list)
        is_homogeneous = l_p > 0.05
        st.write(f"- P-value: **{l_p:.4f}** ({'✅ 등분산' if is_homogeneous else '❌ 이분산'})")

    st.markdown("#### 4. NOEC/LOEC 도출")
    noec, loec = 0, None
    comparisons = []
    
    if num_groups == 2: 
        t_stat, t_p = stats.ttest_ind(control_group, groups[concentrations[1]], equal_var=is_homogeneous)
        if t_p < 0.05: loec = concentrations[1]
        else: noec = concentrations[1]
        st.write(f"T-test P-value: {t_p:.4f}")
    else: 
        if is_normal:
            f_stat, f_p = stats.f_oneway(*data_list)
            if f_p < 0.05:
                alpha = 0.05 / (len(concentrations) - 1) 
                for c in concentrations[1:]:
                    t_s, t_p = stats.ttest_ind(control_group, groups[c], equal_var=is_homogeneous)
                    sig = '🚨 유의' if t_p < alpha else '✅ 차이없음'
                    comparisons.append({'농도': c, 'P-value': f"{t_p:.4f}", '결과': sig})
                    if t_p < alpha and loec is None: loec = c
                    if t_p >= alpha: noec = c
            else:
                noec = max(concentrations)
        else: 
            k_stat, k_p = stats.kruskal(*data_list)
            if k_p < 0.05:
                alpha = 0.05 / (len(concentrations) - 1)
                for c in concentrations[1:]:
                    u_s, u_p = stats.mannwhitneyu(control_group, groups[c], alternative='two-sided')
                    sig = '🚨 유의' if u_p < alpha else '✅ 차이없음'
                    comparisons.append({'농도': c, 'P-value': f"{u_p:.4f}", '결과': sig})
                    if u_p < alpha and loec is None: loec = c
                    if u_p >= alpha: noec = c
            else:
                noec = max(concentrations)
    
    if comparisons: st.table(pd.DataFrame(comparisons))
    c1, c2 = st.columns(2)
    c1.metric(f"{endpoint_name} NOEC", f"{noec} mg/L")
    c2.metric(f"{endpoint_name} LOEC", f"{loec if loec else '> Max'} mg/L")
    st.divider()

# -----------------------------------------------------------------------------
# [함수 3] ECp/LCp 산출 (GLM Probit -> ICPIN Fallback)
# -----------------------------------------------------------------------------
def calculate_ec_lc_range(df, endpoint_col, control_mean, label, is_animal_test=False):
    dose_resp = df.groupby('농도(mg/L)')[endpoint_col].mean().reset_index()
    dose_resp_probit = dose_resp[dose_resp['농도(mg/L)'] > 0].copy()
    
    max_conc = dose_resp['농도(mg/L)'].max()
    p_values = np.arange(5, 100, 5) / 100 
    ec_lc_results = {'p': [], 'value': [], 'status': [], '95% CI': []}
    
    if is_animal_test:
        total_mean = df.groupby('농도(mg/L)')['총 개체수'].mean()
        total_probit = total_mean[dose_resp_probit['농도(mg/L)']].values
        dose_resp_probit['Inhibition'] = dose_resp_probit[endpoint_col] / total_probit
    else:
        dose_resp_probit['Inhibition'] = (control_mean - dose_resp_probit[endpoint_col]) / control_mean

    method_used = "Linear Interpolation (ICp)"
    r_squared = 0
    plot_info = {}
    ci_50_str = "N/C"

    # 1순위: GLM Probit Analysis
    try:
        df_glm = df[df['농도(mg/L)'] > 0].copy()
        
        if is_animal_test:
            df_glm['Log_Conc'] = np.log10(df_glm['농도(mg/L)'])
            grouped = df_glm.groupby('농도(mg/L)').agg(
                Response=(endpoint_col, 'sum'), Total=('총 개체수', 'sum'), Log_Conc=('Log_Conc', 'mean')
            ).reset_index()
            
            grouped.loc[grouped['Response']==grouped['Total'], 'Response'] = grouped['Total'] * 0.999
            grouped.loc[grouped['Response']==0, 'Response'] = grouped['Total'] * 0.001
            
            if grouped['Response'].sum() <= 0: raise ValueError("No response")

            model = sm.GLM(grouped['Response'], sm.add_constant(grouped['Log_Conc']),
                           family=families.Binomial(), exposure=grouped['Total']).fit(disp=False)
            
            intercept, slope = model.params['const'], model.params['Log_Conc']
            pred = model.predict()
            actual = grouped['Response']/grouped['Total']
            r_squared = np.corrcoef(actual, pred)[0,1]**2 if len(actual)>1 else 0

        else:
            df_p = dose_resp_probit.copy()
            df_p['Log_Conc'] = np.log10(df_p['농도(mg/L)'])
            df_p['Inh'] = df_p['Inhibition'].clip(0.001, 0.999)
            df_p['Probit'] = stats.norm.ppf(df_p['Inh'])
            
            model = sm.GLM(df_p['Probit'], sm.add_constant(df_p['Log_Conc']),
                           family=families.Gaussian()).fit(disp=False)
            intercept, slope = model.params['const'], model.params['Log_Conc']
            r_squared = np.corrcoef(df_p['Log_Conc'], df_p['Probit'])[0,1]**2

        if r_squared < 0.6 or slope <= 0: raise ValueError("Low Fit")

        cov = model.cov_params()
        log_lc50 = -intercept / slope
        var_log = (1/slope**2)*(cov.loc['const','const'] + log_lc50**2*cov.loc['Log_Conc','Log_Conc'] + 2*log_lc50*cov.loc['const','Log_Conc'])
        se = np.sqrt(var_log)
        ci_50_str = f"({10**(log_lc50 - 1.96*se):.4f} ~ {10**(log_lc50 + 1.96*se):.4f})"

        for p in p_values:
            ecp = 10**((stats.norm.ppf(p) - intercept)/slope)
            val_s = f"{ecp:.4f}" if 0.05<=p<=0.95 and ecp<max_conc*2 and ecp>0 else "-"
            ec_lc_results['p'].append(int(p*100))
            ec_lc_results['value'].append(val_s)
            ec_lc_results['status'].append("✅ Probit")
            ec_lc_results['95% CI'].append(ci_50_str if int(p*100)==50 else "N/A")

        method_used = "GLM Probit Analysis"
        
        if is_animal_test:
            plot_info = {'type': 'probit', 'x': grouped['Log_Conc'], 'y': stats.norm.ppf(grouped['Response']/grouped['Total']),
                         'slope': slope, 'intercept': intercept, 'r_squared': r_squared,
                         'x_original': grouped['농도(mg/L)'], 'y_original': grouped['Response']/grouped['Total']}
        else:
            plot_info = {'type': 'probit', 'x': df_p['Log_Conc'], 'y': df_p['Probit'],
                         'slope': slope, 'intercept': intercept, 'r_squared': r_squared,
                         'x_original': df_p['농도(mg/L)'], 'y_original': df_p['Inhibition']}

    # 2순위: Linear Interpolation (ICPIN)
    except Exception as e:
        st.warning(f"Probit 모델 실패 ({e}). ICPIN + Bootstrap으로 전환합니다.")
        
        df_icpin = df.copy()
        conc_col = [c for c in df_icpin.columns if '농도' in c][0]
        df_icpin = df_icpin.rename(columns={conc_col: 'Concentration'})
        
        if is_animal_test:
            df_icpin['Value'] = 1 - (df_icpin[endpoint_col] / df_icpin['총 개체수']) 
        else:
            df_icpin['Value'] = df_icpin[endpoint_col] 

        icpin_res, ctrl_val, inh_rates = get_icpin_values_with_ci(df_icpin, 'Value')
        
        method_used = "Linear Interpolation (ICPIN/Bootstrap)"
        ci_50_str = icpin_res['EC50']['lcl']
        ec50_val = icpin_res['EC50']['val']
        
        ec_lc_results = {'p': [], 'value': [], 'status': [], '95% CI': []}
        for p in p_values:
            lvl = int(p*100)
            r = icpin_res.get(f'EC{lvl}', {'val': 'n/a', 'lcl': 'n/a'})
            ec_lc_results['p'].append(lvl)
            ec_lc_results['value'].append(r['val'])
            ec_lc_results['status'].append("✅ Interpol")
            ec_lc_results['95% CI'].append(r['lcl'])
            
        unique_concs = sorted(df_icpin['Concentration'].unique())
        plot_info = {'type': 'linear', 'data': dose_resp, 'r_squared': 0, 
                     'x_original': unique_concs, 
                     'y_original': inh_rates}

    return ec_lc_results, r_squared, method_used, plot_info

# -----------------------------------------------------------------------------
# [함수 4] 그래프 출력 (Dose-Response)
# -----------------------------------------------------------------------------
def plot_ec_lc_curve(plot_info, label, ec_lc_results, y_label="Response (%)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if plot_info['type'] == 'probit':
        x_orig = plot_info['x_original']
        y_orig = plot_info['y_original']
        
        ax.scatter(x_orig, y_orig * 100, color='blue', label='Observed')
        
        x_pred = np.linspace(min(x_orig[x_orig>0]), max(x_orig), 100)
        y_pred = stats.norm.cdf(plot_info['slope']*np.log10(x_pred)+plot_info['intercept']) * 100
        
        ax.plot(x_pred, y_pred, 'r-', label='Probit Fit')
        ax.set_xscale('log') 
        
    else:
        x = plot_info['x_original']
        y = plot_info['y_original']
        ax.plot(x, y*100, 'bo-', label='Observed')
    
    # EC50 Line
    ec50_entry = [res for res in ec_lc_results['value'] if ec_lc_results['p'][ec_lc_results['value'].index(res)] == 50]
    ec50_val = ec50_entry[0] if ec50_entry and ec50_entry[0] != '-' and '>' not in str(ec50_entry[0]) else None
    
    if ec50_val:
        try:
            val = float(ec50_val)
            ax.axvline(val, color='green', linestyle='--', label=f'LC50/EC50: {val}')
        except: pass

    ax.axhline(50, color='gray', linestyle=':')
    ax.set_title(f'{label} Curve')
    ax.set_xlabel('Concentration (mg/L)')
    ax.set_ylabel(y_label)
    ax.legend()
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# [함수 5] 조류 생장 곡선
# -----------------------------------------------------------------------------
def plot_growth_curves(df):
    st.subheader("📈 생장 곡선 (Growth Curves)")
    time_cols = ['0h', '24h', '48h', '72h']
    fig, ax = plt.subplots(figsize=(10, 6))
    concs = sorted(df['농도(mg/L)'].unique())
    for conc in concs:
        subset = df[df['농도(mg/L)'] == conc]
        means = [subset[col].mean() for col in time_cols]
        ax.plot([0, 24, 48, 72], means, marker='o', label=f"{conc} mg/L")
    ax.set_yscale('log')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Cell Density (Log Scale)')
    ax.legend()
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# [분석 실행] 조류 (Algae)
# -----------------------------------------------------------------------------
def run_algae_analysis():
    st.header("🟢 조류 성장저해 시험")
    
    if 'algae_data_full' not in st.session_state:
        st.session_state.algae_data_full = pd.DataFrame({
            '농도(mg/L)': [0]*3 + [10]*3 + [100]*3,
            '0h': [10000]*9,
            '24h': [20000, 21000, 19000, 15000, 16000, 14000, 10000, 10000, 10000],
            '48h': [80000, 82000, 78000, 40000, 42000, 38000, 10000, 10000, 10000],
            '72h': [500000, 510000, 490000, 150000, 160000, 140000, 10000, 10000, 10000]
        })
    
    with st.expander("⚙️ 데이터 입력", expanded=True):
        df_input = st.data_editor(st.session_state.algae_data_full, num_rows="dynamic", use_container_width=True)

    if st.button("분석 실행"):
        df = df_input.copy()
        plot_growth_curves(df)
        st.divider()
        
        init_cells = df['0h'].mean()
        duration = 72
        df['수율'] = df['72h'] - df['0h']
        df['비성장률'] = (np.log(df['72h']) - np.log(df['0h'])) / (duration/24)
        
        c_yield = df[df['농도(mg/L)']==0]['수율'].mean()
        c_rate = df[df['농도(mg/L)']==0]['비성장률'].mean()
        
        tab1, tab2 = st.tabs(["비성장률 (Growth Rate)", "수율 (Yield)"])
        
        with tab1:
            perform_detailed_stats(df, '비성장률', '비성장률')
            res, r2, met, pi = calculate_ec_lc_range(df, '비성장률', c_rate, 'ErC', False)
            idx = [i for i, p in enumerate(res['p']) if p==50][0]
            st.metric("ErC50", f"**{res['value'][idx]} mg/L**", f"95% CI: {res['95% CI'][idx]}")
            st.metric("Model", met)
            st.dataframe(pd.DataFrame(res))
            plot_ec_lc_curve(pi, 'ErC', res)

        with tab2:
            perform_detailed_stats(df, '수율', '수율')
            res, r2, met, pi = calculate_ec_lc_range(df, '수율', c_yield, 'EyC', False)
            idx = [i for i, p in enumerate(res['p']) if p==50][0]
            st.metric("EyC50", f"**{res['value'][idx]} mg/L**", f"95% CI: {res['95% CI'][idx]}")
            st.metric("Model", met)
            st.dataframe(pd.DataFrame(res))
            plot_ec_lc_curve(pi, 'EyC', res)

# -----------------------------------------------------------------------------
# [분석 실행] 물벼룩 (Daphnia) - 24h, 48h
# -----------------------------------------------------------------------------
def run_daphnia_analysis():
    st.header("🦐 물벼룩 급성 유영저해 시험")
    
    if 'daphnia_data' not in st.session_state:
        st.session_state.daphnia_data = pd.DataFrame({
            '농도(mg/L)': [0.0, 6.25, 12.5, 25.0, 50.0, 100.0],
            '총 개체수': [20]*6,
            '반응 수 (24h)': [0, 0, 0, 0, 0, 0],
            '반응 수 (48h)': [0, 0, 1, 5, 18, 20]
        })
    
    df_input = st.data_editor(st.session_state.daphnia_data, num_rows="dynamic", use_container_width=True)
    
    if st.button("상세 분석 실행"):
        df = df_input.copy()
        t24, t48 = st.tabs(["24h 분석", "48h 분석"])
        
        with t24:
            st.subheader("24h EC50 분석")
            ec_res, r2, met, pi = calculate_ec_lc_range(df, '반응 수 (24h)', 0, 'EC', True)
            idx = [i for i, p in enumerate(ec_res['p']) if p==50][0]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("24h EC50", f"**{ec_res['value'][idx]} mg/L**")
            c2.metric("95% CI", ec_res['95% CI'][idx])
            c3.metric("Model", met)
            
            st.dataframe(pd.DataFrame(ec_res).style.apply(lambda x: ['background-color: #e6f3ff']*len(x) if x[f'EC (p)']==50 else ['']*len(x), axis=1))
            plot_ec_lc_curve(pi, "24h EC", ec_res, "Immobility (%)")

        with t48:
            st.subheader("48h EC50 분석")
            ec_res, r2, met, pi = calculate_ec_lc_range(df, '반응 수 (48h)', 0, 'EC', True)
            idx = [i for i, p in enumerate(ec_res['p']) if p==50][0]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("48h EC50", f"**{ec_res['value'][idx]} mg/L**")
            c2.metric("95% CI", ec_res['95% CI'][idx])
            c3.metric("Model", met)
            
            st.dataframe(pd.DataFrame(ec_res).style.apply(lambda x: ['background-color: #e6f3ff']*len(x) if x[f'EC (p)']==50 else ['']*len(x), axis=1))
            plot_ec_lc_curve(pi, "48h EC", ec_res, "Immobility (%)")

# -----------------------------------------------------------------------------
# [분석 실행] 어류 (Fish) - 24h, 48h, 72h, 96h + Slope
# -----------------------------------------------------------------------------
def run_fish_analysis():
    st.header("🐟 어류 급성 독성 시험")
    
    if 'fish_data' not in st.session_state:
        st.session_state.fish_data = pd.DataFrame({
            '농도(mg/L)': [0.0, 6.25, 12.5, 25.0, 50.0, 100.0],
            '총 개체수': [10]*6,
            '반응 수 (24h)': [0]*6,
            '반응 수 (48h)': [0]*6,
            '반응 수 (72h)': [0, 0, 0, 2, 5, 8],
            '반응 수 (96h)': [0, 0, 1, 4, 8, 10]
        })
    
    df_input = st.data_editor(st.session_state.fish_data, num_rows="dynamic", use_container_width=True)
    
    if st.button("상세 분석 실행"):
        df = df_input.copy()
        tabs = st.tabs(["24h", "48h", "72h", "96h (Final)"])
        times = ['24h', '48h', '72h', '96h']
        
        for i, t in enumerate(times):
            with tabs[i]:
                col_name = f'반응 수 ({t})'
                st.subheader(f"{t} LC50 분석")
                
                ec_res, r2, met, pi = calculate_ec_lc_range(df, col_name, 0, 'LC', True)
                idx = [i for i, p in enumerate(ec_res['p']) if p==50][0]
                
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{t} LC50", f"**{ec_res['value'][idx]} mg/L**")
                c2.metric("95% CI", ec_res['95% CI'][idx])
                c3.metric("Model", met)
                
                if t == '96h' and met == 'GLM Probit Analysis':
                    slope_val = pi.get('slope', None)
                    if slope_val:
                        st.info(f"📐 **96h Slope:** {slope_val:.4f}")
                
                st.dataframe(pd.DataFrame(ec_res).style.apply(lambda x: ['background-color: #e6f3ff']*len(x) if x[f'LC (p)']==50 else ['']*len(x), axis=1))
                
                y_lab = "Lethality (%)" if t == '96h' else "Response (%)"
                title_lab = f"{t} Concentration-Lethality" if t == '96h' else f"{t} LC"
                plot_ec_lc_curve(pi, title_lab, ec_res, y_lab)

if __name__ == "__main__":
    if "조류" in analysis_type: run_algae_analysis()
    elif "물벼룩" in analysis_type: run_daphnia_analysis()
    elif "어류" in analysis_type: run_fish_analysis()
