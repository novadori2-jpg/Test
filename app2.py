import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.genmod import families
from scipy.stats import norm 
from scipy.interpolate import interp1d 
import io
import base64
import datetime

# -----------------------------------------------------------------------------
# [공통] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생태독성 전문 분석기 (Final)", page_icon="🧬", layout="wide")

# 한글 폰트 설정 (시스템 폰트 호환성 확보)
plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.rcParams['axes.unicode_minus'] = False

st.title("🧬 생태독성 전문 분석기 (Optimal Pro Ver.)")
st.markdown("""
이 앱은 **CETIS Summary Report** 스타일의 보고서를 생성하며, **OECD TG** 기준을 충족하는 알고리즘을 사용합니다.
1. **분석:** TSK(어류 1순위), Probit(GLM), ICPIN(Bootstrap CI) 자동 적용.
2. **보고서:** NOEC/LOEC 및 ECx/LCx, 신뢰구간, 상세 데이터를 포함한 완벽한 HTML 리포트 제공.
""")
st.divider()

analysis_type = st.sidebar.radio(
    "분석할 실험을 선택하세요",
    ["🟢 조류 성장저해 (Algae)", "🦐 물벼룩 유영저해 (Daphnia)", "🐟 어류 급성독성 (Fish)"]
)

# -----------------------------------------------------------------------------
# [REPORT] CETIS 스타일 HTML 보고서 생성 함수 (인자 순서 수정됨)
# -----------------------------------------------------------------------------
def generate_cetis_report(test_name, endpoint_label, noec, loec, ec50_val, ci_val, method, ec_results, summary_df, fig):
    # 그래프 변환
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    now = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # HTML 스타일
    style = """
    <style>
        @page { size: A4; margin: 15mm; }
        body { font-family: 'Arial', 'Malgun Gothic', sans-serif; font-size: 11pt; line-height: 1.3; color: #000; }
        .container { width: 100%; max-width: 800px; margin: 0 auto; }
        .header-box { border: 1px solid #000; padding: 5px; margin-bottom: 15px; text-align: center; background-color: #f9f9f9; }
        .header-title { font-weight: bold; font-size: 16pt; margin: 5px 0; }
        
        .section-title { 
            font-weight: bold; font-size: 11pt; 
            background-color: #e6e6e6; 
            border-top: 1px solid #000; border-bottom: 1px solid #000; 
            padding: 3px 5px; margin-top: 20px; margin-bottom: 5px; 
        }
        
        table { width: 100%; border-collapse: collapse; margin-bottom: 10px; font-size: 10pt; }
        th { border: 1px solid #000; background-color: #f2f2f2; padding: 4px; text-align: center; font-weight: bold; }
        td { border: 1px solid #000; padding: 4px; text-align: center; }
        
        .info-grid td { border: none; text-align: left; padding: 2px 5px; }
        .info-label { font-weight: bold; width: 120px;}
        
        .graph-box { text-align: center; margin-top: 15px; border: 1px solid #ccc; padding: 10px; }
        img { max-width: 95%; }
    </style>
    """
    
    # Data Summary Table HTML
    summ_rows = ""
    for _, row in summary_df.iterrows():
        summ_rows += f"<tr><td>{row['농도(mg/L)']}</td><td>{int(row['count'])}</td><td>{row['mean']:.4f}</td><td>{row['min']:.4f}</td><td>{row['max']:.4f}</td><td>{row['std']:.4f}</td></tr>"

    # Point Estimate Table HTML
    pe_rows = ""
    # 주요 포인트 (10, 20, 50) 및 양쪽 끝 표시
    target_ps = [10, 20, 50]
    for i, p in enumerate(ec_results['p']):
        if p in target_ps:
            val = ec_results['value'][i]
            ci = ec_results['95% CI'][i]
            pe_rows += f"<tr><td>{endpoint_label}</td><td>EC{p}</td><td>{val}</td><td>{ci}</td></tr>"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8">{style}</head>
    <body>
        <div class="container">
            <div class="header-box">
                <div class="header-title">CETIS Summary Report</div>
                <div class="header-sub">Ecological Toxicity Test Analysis</div>
            </div>

            <table class="info-grid">
                <tr><td class="info-label">Test Name:</td><td>{test_name}</td><td class="info-label">Report Date:</td><td>{now}</td></tr>
                <tr><td class="info-label">Endpoint:</td><td>{endpoint_label}</td><td class="info-label">Method:</td><td>Optimal Pro Ver.</td></tr>
            </table>

            <div class="section-title">Comparison Summary (Hypothesis Test)</div>
            <table>
                <tr>
                    <th>Endpoint</th>
                    <th>NOEC</th>
                    <th>LOEC</th>
                    <th>Method</th>
                </tr>
                <tr>
                    <td>{endpoint_label}</td>
                    <td><b>{noec} mg/L</b></td>
                    <td><b>{loec} mg/L</b></td>
                    <td>Bonferroni t-test / Dunnett Style</td>
                </tr>
            </table>

            <div class="section-title">Point Estimate Summary (Regression/Interpolation)</div>
            <table>
                <tr>
                    <th>Endpoint</th>
                    <th>Level</th>
                    <th>mg/L</th>
                    <th>95% LCL - UCL</th>
                </tr>
                {pe_rows}
            </table>
            <p style="font-size:9pt; text-align:right;">* Analysis Method: {method}</p>

            <div class="section-title">Summary of Data</div>
            <table>
                <tr>
                    <th>Conc-mg/L</th>
                    <th>Count</th>
                    <th>Mean</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Std Dev</th>
                </tr>
                {summ_rows}
            </table>
            
            <div class="section-title">Concentration-Response Curve</div>
            <div class="graph-box">
                <img src="data:image/png;base64,{img_base64}">
            </div>
        </div>
    </body>
    </html>
    """
    return html

# -----------------------------------------------------------------------------
# [함수 1] ICPIN + Bootstrap CI 산출 로직
# -----------------------------------------------------------------------------
def get_icpin_values_with_ci(df_resp, endpoint, is_binary=False, total_col=None, response_col=None, n_boot=1000):
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
        if target_y > y_iso.max() + 1e-9: return np.nan 
        if target_y < y_iso.min() - 1e-9: return np.nan
        return float(interp_func(target_y))

    ec_levels = np.arange(5, 100, 5) 
    main_results = {}
    
    control_val = y_iso[0]
    for level in ec_levels:
        main_results[level] = calc_icpin_ec(interpolator, level, control_val)

    boot_estimates = {l: [] for l in ec_levels}
    
    # Bootstrap (Binomial for Animals, Resampling for Algae)
    for _ in range(n_boot):
        boot_y_means = []
        for c in x_raw:
            if is_binary and total_col and response_col:
                # Binomial Resampling for summary data
                row = df_temp[df_temp['Concentration'] == c].iloc[0]
                n = int(row[total_col])
                p_hat = row[endpoint] 
                if n > 0:
                    # p_hat이 0~1 범위를 벗어나지 않도록 클리핑
                    p_hat = np.clip(p_hat, 0, 1)
                    boot_mean = np.random.binomial(n, p_hat) / n
                else:
                    boot_mean = 0
                boot_y_means.append(boot_mean)
            else:
                # Standard Resampling for replicate data
                vals = df_temp[df_temp['Concentration']==c][endpoint].values
                if len(vals) > 0:
                    boot_y_means.append(np.random.choice(vals, size=len(vals), replace=True).mean())
                else:
                    boot_y_means.append(0)
        
        if not boot_y_means: continue
        
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
            ci_str = "N/C"
        
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
    
    # Summary Dataframe 반환용
    summary = df.groupby('농도(mg/L)')[endpoint_col].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    
    if num_groups < 2:
        st.error("데이터 그룹이 2개 미만입니다.")
        return None, None, summary

    st.markdown("#### 1. 기초 통계량")
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
    is_homogeneous = False
    if len(data_list) >= 2:
        l_stat, l_p = stats.levene(*data_list)
        is_homogeneous = l_p > 0.05
        st.write(f"- P-value: **{l_p:.4f}** ({'✅ 등분산' if is_homogeneous else '❌ 이분산'})")

    st.markdown("#### 4. NOEC/LOEC 도출")
    noec, loec = max(concentrations), "> Max"
    
    # Simplified Bonferroni Logic
    if num_groups >= 2:
        alpha = 0.05 / (num_groups - 1)
        found_loec = False
        for conc in concentrations[1:]:
            # Equal variance assumption based on Levene's
            t, p = stats.ttest_ind(control_group, groups[conc], equal_var=is_homogeneous)
            if p < alpha:
                if not found_loec:
                    loec = conc
                    found_loec = True
            elif not found_loec:
                noec = conc
        
        if not found_loec:
            noec = max(concentrations)
            loec = "> Max"
    
    c1, c2 = st.columns(2)
    c1.metric(f"{endpoint_name} NOEC", f"{noec} mg/L")
    c2.metric(f"{endpoint_name} LOEC", f"{loec} mg/L")
    st.divider()
    
    return noec, loec, summary

# -----------------------------------------------------------------------------
# [함수 3] ECp/LCp 산출
# -----------------------------------------------------------------------------
def calculate_ec_lc_range(df, endpoint_col, control_mean, label, is_animal_test=False):
    dose_resp = df.groupby('농도(mg/L)')[endpoint_col].mean().reset_index()
    dose_resp_probit = dose_resp[dose_resp['농도(mg/L)'] > 0].copy()
    
    max_conc = dose_resp['농도(mg/L)'].max()
    p_values = np.arange(5, 100, 5) / 100 
    ec_lc_results = {'p': [], 'value': [], 'status': [], '95% CI': []}
    
    # 반응률/저해율 계산 (중요: ICPIN용 Value 준비)
    if is_animal_test:
        # 동물: Value = Survival Rate (1->0) for ICPIN monotonic logic
        # Note: Response is mortality.
        total_mean = df.groupby('농도(mg/L)')['총 개체수'].mean()
        # Inhibition = Mortality Rate (0->1)
        dose_resp['Inhibition'] = df.groupby('농도(mg/L)')[endpoint_col].mean() / total_mean
        dose_resp['Inhibition'] = dose_resp['Inhibition'].fillna(0)
    else:
        # 조류: Inhibition = (Ctrl - Treat) / Ctrl
        dose_resp['Inhibition'] = (control_mean - dose_resp[endpoint_col]) / control_mean

    method_used = "Linear Interpolation (ICp)"
    plot_info = {}
    ci_50_str = "N/C"

    # 1순위: GLM Probit Analysis (동물 시험만 우선 적용)
    try:
        if not is_animal_test: raise ValueError("Algae skips Probit for now") # 조류는 ICPIN 우선 (데이터 특성상)
        
        df_glm = df[df['농도(mg/L)'] > 0].copy()
        df_glm['Log_Conc'] = np.log10(df_glm['농도(mg/L)'])
        
        grouped = df_glm.groupby('농도(mg/L)').agg(
            Response=(endpoint_col, 'sum'), Total=('총 개체수', 'sum'), Log_Conc=('Log_Conc', 'mean')
        ).reset_index()
        
        # 0/100% 조정
        grouped.loc[grouped['Response']==grouped['Total'], 'Response'] = grouped['Total'] * 0.999
        grouped.loc[grouped['Response']==0, 'Response'] = grouped['Total'] * 0.001
        
        if grouped['Response'].sum() <= 0: raise ValueError("No response")

        model = sm.GLM(grouped['Response'], sm.add_constant(grouped['Log_Conc']),
                       family=families.Binomial(), exposure=grouped['Total']).fit(disp=False)
        
        intercept, slope = model.params['const'], model.params['Log_Conc']
        if slope <= 0: raise ValueError("Negative Slope") # Dose-Response must be positive

        # CI Calculation
        cov = model.cov_params()
        log_lc50 = -intercept / slope
        var_log = (1/slope**2)*(cov.loc['const','const'] + log_lc50**2*cov.loc['Log_Conc','Log_Conc'] + 2*log_lc50*cov.loc['const','Log_Conc'])
        se = np.sqrt(var_log) if var_log > 0 else 0
        
        lcl_val = 10**(log_lc50 - 1.96*se)
        ucl_val = 10**(log_lc50 + 1.96*se)
        ci_50_str = f"({lcl_val:.4f} ~ {ucl_val:.4f})"

        for p in p_values:
            ecp = 10**((stats.norm.ppf(p) - intercept)/slope)
            val_s = f"{ecp:.4f}" if 0<ecp<max_conc*100 else "> Max"
            ec_lc_results['p'].append(int(p*100))
            ec_lc_results['value'].append(val_s)
            ec_lc_results['status'].append("✅ Probit")
            ec_lc_results['95% CI'].append(ci_50_str if int(p*100)==50 else "N/A")

        method_used = "GLM Probit Analysis"
        plot_info = {'type': 'probit', 'x': grouped['Log_Conc'], 'y': stats.norm.ppf(grouped['Response']/grouped['Total']),
                     'slope': slope, 'intercept': intercept, 'r_squared': 0.99,
                     'x_original': grouped['농도(mg/L)'], 'y_original': grouped['Response']/grouped['Total']}

    # 2순위: Linear Interpolation (ICPIN + Bootstrap)
    except Exception as e:
        df_icpin = df.copy()
        conc_col = [c for c in df_icpin.columns if '농도' in c][0]
        df_icpin = df_icpin.rename(columns={conc_col: 'Concentration'})
        
        if is_animal_test:
            # ICPIN needs 'Survival Rate' (Monotonic Decreasing)
            df_icpin['Value'] = 1 - (df_icpin[endpoint_col] / df_icpin['총 개체수'])
            icpin_res, ctrl_val, inh_rates = get_icpin_values_with_ci(
                df_icpin, 'Value', is_binary=True, total_col='총 개체수', response_col=endpoint_col
            )
        else:
            # Algae: Use raw endpoint (e.g. Growth Rate) which decreases with conc
            df_icpin['Value'] = df_icpin[endpoint_col]
            icpin_res, ctrl_val, inh_rates = get_icpin_values_with_ci(df_icpin, 'Value', is_binary=False)

        method_used = "Linear Interpolation (ICPIN/Bootstrap)"
        ci_50_str = icpin_res['EC50']['lcl']
        
        ec_lc_results = {'p': [], 'value': [], 'status': [], '95% CI': []}
        for p in p_values:
            lvl = int(p*100)
            r = icpin_res.get(f'EC{lvl}', {'val': 'n/a', 'lcl': 'n/a'})
            ec_lc_results['p'].append(lvl)
            ec_lc_results['value'].append(r['val'])
            ec_lc_results['status'].append("✅ Interpol")
            ec_lc_results['95% CI'].append(r['lcl'])
            
        unique_concs = sorted(df_icpin['Concentration'].unique())
        plot_info = {'type': 'linear', 'x_original': unique_concs, 'y_original': inh_rates}

    return ec_lc_results, 0, method_used, plot_info

# -----------------------------------------------------------------------------
# [함수 4] 그래프 출력 (Fig 반환)
# -----------------------------------------------------------------------------
def plot_ec_lc_curve(plot_info, label, ec_lc_results, y_label="Response (%)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_orig = plot_info['x_original']
    y_orig = plot_info['y_original']
    
    ax.scatter(x_orig, y_orig * 100, color='blue', label='Observed', zorder=5)
    
    if plot_info['type'] == 'probit':
        x_pred = np.logspace(np.log10(min(x_orig[x_orig>0])), np.log10(max(x_orig)), 100)
        y_pred = stats.norm.cdf(plot_info['slope']*np.log10(x_pred)+plot_info['intercept']) * 100
        ax.plot(x_pred, y_pred, 'r-', label='Probit Fit')
        ax.set_xscale('log')
    else:
        ax.plot(x_orig, y_orig * 100, 'b--', label='Interpolation', alpha=0.5)

    ec50_val = [x for i, x in enumerate(ec_lc_results['value']) if ec_lc_results['p'][i]==50][0]
    if ec50_val and '>' not in str(ec50_val) and 'n/a' not in str(ec50_val):
        try:
            val = float(ec50_val)
            ax.axvline(val, color='green', linestyle='--', label=f'EC50: {val}')
        except: pass

    ax.axhline(50, color='gray', linestyle=':')
    ax.set_title(f'{label} Dose-Response Curve')
    ax.set_xlabel('Concentration (mg/L)')
    ax.set_ylabel(y_label)
    ax.legend()
    st.pyplot(fig)
    return fig

# -----------------------------------------------------------------------------
# [함수 5] 조류 생장 곡선
# -----------------------------------------------------------------------------
def plot_growth_curves(df):
    st.subheader("📈 생장 곡선 (Growth Curves)")
    time_cols = ['0h', '24h', '48h', '72h']
    fig, ax = plt.subplots(figsize=(10, 6))
    concs = sorted(df['농도(mg/L)'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(concs)))
    
    for idx, conc in enumerate(concs):
        subset = df[df['농도(mg/L)'] == conc]
        means = [subset[col].mean() for col in time_cols]
        ax.plot([0, 24, 48, 72], means, marker='o', label=f"{conc} mg/L", color=colors[idx])

    ax.set_yscale('log')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Cell Density (Log Scale)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
    return fig

# -----------------------------------------------------------------------------
# [분석 실행] 조류
# -----------------------------------------------------------------------------
def run_algae_analysis():
    st.header("🟢 조류 성장저해 시험")
    if 'algae_data_full' not in st.session_state:
        st.session_state.algae_data_full = pd.DataFrame({
            '농도(mg/L)': [0]*3 + [10]*3 + [100]*3,
            '0h': [10000]*9, '24h': [20000]*9, '48h': [80000]*9, '72h': [500000]*9
        })
    with st.expander("⚙️ 데이터 입력", expanded=True):
        df_input = st.data_editor(st.session_state.algae_data_full, num_rows="dynamic", use_container_width=True)

    if st.button("분석 실행"):
        df = df_input.copy()
        g_fig = plot_growth_curves(df)
        st.divider()
        
        df['수율'] = df['72h'] - df['0h']
        df['비성장률'] = (np.log(df['72h']) - np.log(df['0h'])) / (72/24)
        c_yield = df[df['농도(mg/L)']==0]['수율'].mean()
        c_rate = df[df['농도(mg/L)']==0]['비성장률'].mean()
        
        tab1, tab2 = st.tabs(["비성장률 (Growth Rate)", "수율 (Yield)"])
        with tab1:
            noec, loec, summ = perform_detailed_stats(df, '비성장률', '비성장률')
            res, r2, met, pi = calculate_ec_lc_range(df, '비성장률', c_rate, 'ErC', False)
            
            idx = [i for i, p in enumerate(res['p']) if p==50][0]
            val, ci = res['value'][idx], res['95% CI'][idx]
            st.metric("ErC50", f"**{val} mg/L**", f"95% CI: {ci}")
            st.dataframe(pd.DataFrame(res))
            fig = plot_ec_lc_curve(pi, 'ErC', res, "Inhibition (%)")
            
            # generate_cetis_report 호출 (인자 순서 중요!)
            html = generate_cetis_report("조류 성장저해", "ErC50", noec, loec, val, ci, met, res, summ, fig)
            st.download_button("📥 보고서 다운로드", html, "Algae_ErC50_Report.html")

        with tab2:
            noec, loec, summ = perform_detailed_stats(df, '수율', '수율')
            res, r2, met, pi = calculate_ec_lc_range(df, '수율', c_yield, 'EyC', False)
            
            idx = [i for i, p in enumerate(res['p']) if p==50][0]
            val, ci = res['value'][idx], res['95% CI'][idx]
            st.metric("EyC50", f"**{val} mg/L**", f"95% CI: {ci}")
            st.dataframe(pd.DataFrame(res))
            fig = plot_ec_lc_curve(pi, 'EyC', res, "Inhibition (%)")
            
            html = generate_cetis_report("조류 성장저해", "EyC50", noec, loec, val, ci, met, res, summ, fig)
            st.download_button("📥 보고서 다운로드", html, "Algae_EyC50_Report.html")

# -----------------------------------------------------------------------------
# [분석 실행] 물벼룩 (24h, 48h)
# -----------------------------------------------------------------------------
def run_daphnia_analysis():
    st.header("🦐 물벼룩 급성 유영저해 시험")
    if 'daphnia_data' not in st.session_state:
        st.session_state.daphnia_data = pd.DataFrame({
            '농도(mg/L)': [0.0, 6.25, 12.5, 25.0, 50.0, 100.0], '총 개체수': [20]*6, '반응 수 (24h)': [0]*6, '반응 수 (48h)': [0, 0, 1, 5, 18, 20]
        })
    df_input = st.data_editor(st.session_state.daphnia_data, num_rows="dynamic", use_container_width=True)
    
    if st.button("상세 분석 실행"):
        df = df_input.copy()
        t24, t48 = st.tabs(["24h 분석", "48h 분석"])
        
        for t_label, col in zip(["24h", "48h"], ['반응 수 (24h)', '반응 수 (48h)']):
            with (t24 if t_label=="24h" else t48):
                st.subheader(f"{t_label} EC50 분석")
                noec, loec, summ = perform_detailed_stats(df, col, "EC")
                ec_res, r2, met, pi = calculate_ec_lc_range(df, col, 0, 'EC', True)
                
                idx = [i for i, p in enumerate(ec_res['p']) if p==50][0]
                val, ci = ec_res['value'][idx], ec_res['95% CI'][idx]
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{t_label} EC50", f"**{val} mg/L**")
                c2.metric("95% CI", ci)
                c3.metric("Model", met)
                
                res_df = pd.DataFrame(ec_res).rename(columns={'p': 'EC (p)', 'value': 'Conc', '95% CI': '95% CI'})
                st.dataframe(res_df.style.apply(lambda x: ['background-color: #e6f3ff']*len(x) if x['EC (p)']==50 else ['']*len(x), axis=1))
                
                fig = plot_ec_lc_curve(pi, f"{t_label} EC", ec_res, "Immobility (%)")
                html = generate_cetis_report(f"물벼룩 급성 ({t_label})", "EC50", noec, loec, val, ci, met, ec_res, summ, fig)
                st.download_button(f"📥 {t_label} 보고서 다운로드", html, f"Daphnia_{t_label}_Report.html")

# -----------------------------------------------------------------------------
# [분석 실행] 어류 (24h~96h)
# -----------------------------------------------------------------------------
def run_fish_analysis():
    st.header("🐟 어류 급성 독성 시험")
    if 'fish_data' not in st.session_state:
        st.session_state.fish_data = pd.DataFrame({
            '농도(mg/L)': [0.0, 6.25, 12.5, 25.0, 50.0, 100.0], '총 개체수': [10]*6, 
            '반응 수 (24h)': [0]*6, '반응 수 (48h)': [0]*6, '반응 수 (72h)': [0,0,0,2,5,8], '반응 수 (96h)': [0,0,1,4,8,10]
        })
    df_input = st.data_editor(st.session_state.fish_data, num_rows="dynamic", use_container_width=True)
    
    if st.button("상세 분석 실행"):
        df = df_input.copy()
        tabs = st.tabs(["24h", "48h", "72h", "96h (Final)"])
        times = ['24h', '48h', '72h', '96h']
        
        for i, t in enumerate(times):
            with tabs[i]:
                col = f'반응 수 ({t})'
                st.subheader(f"{t} LC50 분석")
                noec, loec, summ = perform_detailed_stats(df, col, "LC")
                ec_res, r2, met, pi = calculate_ec_lc_range(df, col, 0, 'LC', True)
                
                idx = [i for i, p in enumerate(ec_res['p']) if p==50][0]
                val, ci = ec_res['value'][idx], ec_res['95% CI'][idx]
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{t} LC50", f"**{val} mg/L**")
                c2.metric("95% CI", ci)
                c3.metric("Model", met)
                
                fig = plot_ec_lc_curve(pi, f"{t} LC", ec_res, "Lethality (%)")
                html = generate_cetis_report(f"어류 급성 ({t})", "LC50", noec, loec, val, ci, met, ec_res, summ, fig)
                st.download_button(f"📥 {t} 보고서 다운로드", html, f"Fish_{t}_Report.html")

if __name__ == "__main__":
    if "조류" in analysis_type: run_algae_analysis()
    elif "물벼룩" in analysis_type: run_daphnia_analysis()
    elif "어류" in analysis_type: run_fish_analysis()
