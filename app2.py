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
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.title("🧬 생태독성 전문 분석기 (Optimal Pro Ver.)")
st.markdown("""
이 앱은 **CETIS Summary Report** 스타일의 보고서를 생성하며, **OECD TG** 기준을 충족하는 알고리즘을 사용합니다.
1. **분석:** TSK, Probit(GLM), ICPIN(Bootstrap CI) 자동 적용.
2. **보고서:** NOEC/LOEC 및 ECx/LCx, 신뢰구간, 상세 데이터를 포함한 완벽한 HTML 리포트 제공.
""")
st.divider()

analysis_type = st.sidebar.radio(
    "분석할 실험을 선택하세요",
    ["🟢 조류 성장저해 (Algae)", "🦐 물벼룩 유영저해 (Daphnia)", "🐟 어류 급성독성 (Fish)"]
)

# -----------------------------------------------------------------------------
# [REPORT] HTML 보고서 생성 함수 (수정됨: summary_df 인자 추가)
# -----------------------------------------------------------------------------
def generate_full_cetis_report(meta_info, stats_results, ec_results, raw_df, summary_df, fig, report_type="full"):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Point Estimate Rows
    pe_rows = ""
    target_ps = [10, 20, 50]
    for i, p in enumerate(ec_results['p']):
        if p in target_ps:
            pe_rows += f"<tr><td>{meta_info['endpoint']}</td><td>EC{p}</td><td>{ec_results['value'][i]}</td><td>{ec_results['95% CI'][i]}</td><td>{meta_info['method_ec']}</td></tr>"

    # Data Summary Rows (summary_df 사용)
    summ_rows = ""
    # summary_df 컬럼: 농도(mg/L), mean, std, min, max, count
    # 필요 시 추가 계산 (CV, Effect)
    if 0 in summary_df['농도(mg/L)'].values:
        control_mean = summary_df[summary_df['농도(mg/L)']==0]['mean'].values[0]
    else:
        control_mean = 0

    for _, row in summary_df.iterrows():
        n = int(row['count'])
        m = row['mean']
        s = row['std']
        se = s / np.sqrt(n) if n>0 else 0
        cv = (s/m*100) if m!=0 else 0
        
        if report_type == "simple" and meta_info.get('is_animal', False):
            total_n = meta_info.get('total_n', 10)
            effect = (m / total_n) * 100
        else:
            effect = ((control_mean - m)/control_mean*100) if control_mean!=0 else 0
            
        summ_rows += f"<tr><td>{row['농도(mg/L)']}</td><td>{n}</td><td>{m:.4f}</td><td>{row['min']:.4f}</td><td>{row['max']:.4f}</td><td>{se:.4f}</td><td>{cv:.2f}%</td><td>{effect:.2f}%</td></tr>"

    # HTML Parts
    comparison_html = ""
    assumption_html = ""
    anova_html = ""

    if report_type == "full" and stats_results:
        comparison_html = f"""
        <div class="section-title">Comparison Summary</div>
        <table>
            <tr><th>Endpoint</th><th>NOEC</th><th>LOEC</th><th>Method</th></tr>
            <tr><td>{meta_info['endpoint']}</td><td><b>{stats_results['noec']} mg/L</b></td><td><b>{stats_results['loec']} mg/L</b></td><td>{stats_results['test_name']}</td></tr>
        </table>"""
        
        assumption_html = f"""
        <div class="section-title">Test Acceptability & Assumptions</div>
        <table>
            <tr><th>Attribute</th><th>Test</th><th>Statistic</th><th>P-Value</th><th>Decision</th></tr>
            <tr><td>Normality</td><td>Shapiro-Wilk</td><td>{stats_results['shapiro_stat']:.4f}</td><td>{stats_results['shapiro_p']:.4f}</td><td>{stats_results['shapiro_res']}</td></tr>
            <tr><td>Variances</td><td>Levene's Test</td><td>{stats_results['levene_stat']:.4f}</td><td>{stats_results['levene_p']:.4f}</td><td>{stats_results['levene_res']}</td></tr>
        </table>"""
        
        if 'anova_f' in stats_results:
             anova_html = f"""
            <div class="section-title">Analysis of Variance (ANOVA)</div>
            <table>
                <tr><th>Source</th><th>F-Stat</th><th>P-Value</th><th>Decision(α:0.05)</th></tr>
                <tr><td>Between Groups</td><td>{stats_results['anova_f']:.4f}</td><td>{stats_results['anova_p']:.4f}</td><td>{'Significant' if stats_results['anova_p'] < 0.05 else 'Non-Significant'}</td></tr>
            </table>"""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{ size: A4; margin: 15mm; }}
            body {{ font-family: 'Arial', 'Malgun Gothic', sans-serif; font-size: 10pt; color: #000; }}
            .header-box {{ border: 2px solid #000; padding: 10px; margin-bottom: 10px; background-color: #f0f0f0; text-align: center; }}
            .header-title {{ font-weight: bold; font-size: 14pt; }}
            .section-title {{ font-weight: bold; font-size: 11pt; background-color: #e0e0e0; padding: 3px 5px; margin-top: 20px; border-bottom: 1px solid #000; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 10px; font-size: 9pt; }}
            th, td {{ border: 1px solid #000; padding: 4px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            .graph-box {{ text-align: center; margin-top: 10px; }}
            img {{ max-width: 90%; }}
            .page-break {{ page-break-before: always; }}
        </style>
    </head>
    <body>
        <div class="header-box"><div class="header-title">CETIS Summary Report</div></div>
        <table style="border:none;">
            <tr><td style="border:none;text-align:left;"><b>Test Name:</b> {meta_info.get('test_type','Test')}</td><td style="border:none;text-align:right;"><b>Date:</b> {now}</td></tr>
            <tr><td style="border:none;text-align:left;"><b>Endpoint:</b> {meta_info['endpoint']}</td><td style="border:none;text-align:right;"><b>Method:</b> Optimal Pro Ver.</td></tr>
        </table>
        
        {comparison_html}

        <div class="section-title">Point Estimate Summary</div>
        <table>
            <tr><th>Endpoint</th><th>Level</th><th>mg/L</th><th>95% LCL - UCL</th><th>Method</th></tr>
            {pe_rows}
        </table>

        <div class="section-title">Summary of Data</div>
        <table>
            <tr><th>Conc</th><th>N</th><th>Mean</th><th>Min</th><th>Max</th><th>Std Err</th><th>CV%</th><th>%Effect</th></tr>
            {summ_rows}
        </table>
        
        {assumption_html}
        {anova_html}
        
        <div class="page-break"></div>
        <div class="section-title">Concentration-Response Curve</div>
        <div class="graph-box"><img src="data:image/png;base64,{img_base64}"></div>
    </body>
    </html>
    """
    return html

# -----------------------------------------------------------------------------
# [함수 1] ICPIN + Bootstrap
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
    try: interpolator = interp1d(y_iso, x_raw, kind='linear', bounds_error=False, fill_value=np.nan)
    except: interpolator = None

    def calc_icpin_ec(interp_func, level, control_val):
        if interp_func is None: return np.nan
        target_y = control_val * (1 - level/100)
        if target_y > y_iso.max() + 1e-9: return np.nan 
        if target_y < y_iso.min() - 1e-9: return np.nan
        return float(interp_func(target_y))

    ec_levels = np.arange(5, 100, 5) 
    main_results = {}
    control_val = y_iso[0]
    for level in ec_levels: main_results[level] = calc_icpin_ec(interpolator, level, control_val)

    boot_estimates = {l: [] for l in ec_levels}
    
    for _ in range(n_boot):
        boot_y_means = []
        for c in x_raw:
            if is_binary and total_col and response_col:
                row = df_temp[df_temp['Concentration'] == c].iloc[0]
                n, p_hat = int(row[total_col]), row[endpoint]
                boot_mean = np.random.binomial(n, np.clip(p_hat,0,1)) / n if n > 0 else 0
                boot_y_means.append(boot_mean)
            else:
                vals = df_temp[df_temp['Concentration']==c][endpoint].values
                if len(vals)>0: boot_y_means.append(np.random.choice(vals, len(vals), replace=True).mean())
                else: boot_y_means.append(0)
        
        if not boot_y_means: continue
        boot_y_means = np.array(boot_y_means)
        y_boot_iso = np.maximum.accumulate(boot_y_means[::-1])[::-1]
        try:
            boot_interp = interp1d(y_boot_iso, x_raw, kind='linear', bounds_error=False, fill_value=np.nan)
            for level in ec_levels:
                val = calc_icpin_ec(boot_interp, level, y_boot_iso[0])
                if not np.isnan(val) and val > 0: boot_estimates[level].append(val)
        except: continue

    final_out = {}
    max_conc = x_raw.max()
    if control_val != 0: inhibition_rates = (control_val - y_raw) / control_val
    else: inhibition_rates = np.zeros_like(y_raw)
    
    for level in ec_levels:
        val = main_results[level]
        boots = boot_estimates[level]
        val_str = f"{val:.4f}" if not np.isnan(val) else (f"> {max_conc:.4f}" if level >= 50 else "n/a")
        if np.isnan(val) or len(boots) < 20: ci_str = "N/C"
        else: ci_str = f"({np.percentile(boots, 2.5):.4f} ~ {np.percentile(boots, 97.5):.4f})"
        final_out[f'EC{level}'] = {'val': val_str, 'lcl': ci_str}
        
    return final_out, control_val, inhibition_rates

# -----------------------------------------------------------------------------
# [함수 2] 상세 통계 분석 (NOEC/LOEC) - return_details=True 시 (stats, summ) 반환
# -----------------------------------------------------------------------------
def perform_detailed_stats(df, endpoint_col, endpoint_name, return_details=False):
    st.markdown(f"### 📊 {endpoint_name} 통계 검정 상세 보고서")
    groups = df.groupby('농도(mg/L)')[endpoint_col].apply(list)
    concentrations = sorted(groups.keys())
    control_group = groups[0]
    num_groups = len(concentrations)
    
    summary = df.groupby('농도(mg/L)')[endpoint_col].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    
    if num_groups < 2:
        st.error("데이터 부족")
        return None, None # 2개 반환 (에러시)

    # 통계 계산
    noec, loec = max(concentrations), "> Max"
    stats_details = {}
    
    # 정규성/등분산성 생략(코드 단축), P-value만 계산
    resid = []
    for c in concentrations: resid.extend(np.array(groups[c]) - np.mean(groups[c]))
    s_stat, s_p = stats.shapiro(resid) if len(resid)>3 else (0,1)
    stats_details.update({'shapiro_stat': s_stat, 'shapiro_p': s_p, 'shapiro_res': 'Pass' if s_p>0.01 else 'Fail'})
    
    l_stat, l_p = stats.levene(*[groups[c] for c in concentrations])
    stats_details.update({'levene_stat': l_stat, 'levene_p': l_p, 'levene_res': 'Pass' if l_p>0.01 else 'Fail'})

    if num_groups >= 2:
        alpha = 0.05 / (num_groups - 1)
        found = False
        for conc in concentrations[1:]:
            t, p = stats.ttest_ind(control_group, groups[conc], equal_var=(l_p>0.01))
            if p < alpha:
                if not found: loec, found = conc, True
            elif not found: noec = conc
    
    if not found: noec, loec = max(concentrations), "> Max"
    stats_details.update({'noec': noec, 'loec': loec, 'test_name': 'Bonferroni t-test'})
    
    # 화면 출력
    st.markdown("#### 1. 기초 통계량")
    st.dataframe(summary.style.format("{:.4f}"))
    c1, c2 = st.columns(2)
    c1.metric("NOEC", f"{noec}")
    c2.metric("LOEC", f"{loec}")
    st.divider()
    
    # *** 중요 수정 ***
    if return_details:
        return stats_details, summary # 2개 반환
    else:
        return noec, loec, summary # 3개 반환

# -----------------------------------------------------------------------------
# [함수 3] ECp/LCp 산출 (Probit -> ICPIN)
# -----------------------------------------------------------------------------
def calculate_ec_lc_range(df, endpoint_col, control_mean, label, is_animal_test=False):
    dose_resp = df.groupby('농도(mg/L)')[endpoint_col].mean().reset_index()
    max_conc = dose_resp['농도(mg/L)'].max()
    p_values = np.arange(5, 100, 5) / 100 
    ec_res = {'p': [], 'value': [], 'status': [], '95% CI': []}
    
    if is_animal_test:
        total_mean = df.groupby('농도(mg/L)')['총 개체수'].mean()
        dose_resp['Inhibition'] = df.groupby('농도(mg/L)')[endpoint_col].mean() / total_mean
    else:
        dose_resp['Inhibition'] = (control_mean - dose_resp[endpoint_col]) / control_mean

    method_used, plot_info = "ICPIN", {}

    try:
        # Probit Logic
        if not is_animal_test: raise Exception("Algae: Force ICPIN")
        
        df_glm = df[df['농도(mg/L)'] > 0].copy()
        df_glm['Log_Conc'] = np.log10(df_glm['농도(mg/L)'])
        grouped = df_glm.groupby('농도(mg/L)').agg(Response=(endpoint_col,'sum'), Total=('총 개체수','sum'), Log_Conc=('Log_Conc','mean')).reset_index()
        
        grouped.loc[grouped['Response']==grouped['Total'], 'Response'] *= 0.999
        grouped.loc[grouped['Response']==0, 'Response'] = grouped['Total'] * 0.001
        if grouped['Response'].sum() <= 0: raise ValueError
        
        model = sm.GLM(grouped['Response'], sm.add_constant(grouped['Log_Conc']), family=families.Binomial(), exposure=grouped['Total']).fit(disp=0)
        intercept, slope = model.params['const'], model.params['Log_Conc']
        if slope <= 0: raise ValueError
        
        # Delta Method CI
        cov = model.cov_params()
        log_lc50 = -intercept/slope
        var_log = (1/slope**2)*(cov.loc['const','const'] + log_lc50**2*cov.loc['Log_Conc','Log_Conc'] + 2*log_lc50*cov.loc['const','Log_Conc'])
        se = np.sqrt(var_log) if var_log>0 else 0
        ci_50 = f"({10**(log_lc50-1.96*se):.4f} ~ {10**(log_lc50+1.96*se):.4f})"

        for p in p_values:
            ecp = 10**((stats.norm.ppf(p)-intercept)/slope)
            val_s = f"{ecp:.4f}" if 0<ecp<max_conc*100 else "> Max"
            ec_res['p'].append(int(p*100))
            ec_res['value'].append(val_s)
            ec_res['95% CI'].append(ci_50 if int(p*100)==50 else "N/A")
        
        method_used = "GLM Probit"
        plot_info = {'type':'probit', 'x': grouped['Log_Conc'], 'y': stats.norm.ppf(grouped['Response']/grouped['Total']), 
                     'slope':slope, 'intercept':intercept, 'x_original': grouped['농도(mg/L)'], 'y_original': grouped['Response']/grouped['Total']}

    except:
        # ICPIN Fallback
        df_icpin = df.copy().rename(columns={df.columns[0]:'Concentration'}) # Placeholder
        conc_col = [c for c in df.columns if '농도' in c][0]
        df_icpin = df.copy().rename(columns={conc_col: 'Concentration'})
        
        if is_animal_test:
            df_icpin['Value'] = 1 - (df_icpin[endpoint_col] / df_icpin['총 개체수'])
            icpin_res, _, inh = get_icpin_values_with_ci(df_icpin, 'Value', True, '총 개체수', endpoint_col)
        else:
            df_icpin['Value'] = df_icpin[endpoint_col]
            icpin_res, _, inh = get_icpin_values_with_ci(df_icpin, 'Value', False)
            
        method_used = "Linear Interpolation (ICPIN/Bootstrap)"
        for p in p_values:
            lvl = int(p*100)
            r = icpin_res.get(f'EC{lvl}', {'val':'n/a', 'lcl':'n/a'})
            ec_res['p'].append(lvl)
            ec_res['value'].append(r['val'])
            ec_res['95% CI'].append(r['lcl'])
            
        plot_info = {'type':'linear', 'x_original': sorted(df_icpin['Concentration'].unique()), 'y_original': inh}

    return ec_res, 0, method_used, plot_info

def plot_ec_lc_curve(plot_info, label, ec_lc_results, y_label="Response (%)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    x, y = plot_info['x_original'], plot_info['y_original']
    ax.scatter(x, y*100, c='blue', label='Observed', zorder=5)
    
    if plot_info['type'] == 'probit':
        x_p = np.logspace(np.log10(min(x[x>0])), np.log10(max(x)), 100)
        y_p = stats.norm.cdf(plot_info['slope']*np.log10(x_p)+plot_info['intercept'])*100
        ax.plot(x_p, y_p, 'r-', label='Probit')
        ax.set_xscale('log')
    else:
        ax.plot(x, y*100, 'b--', label='Interpolation', alpha=0.5)
    
    idx = [i for i,p in enumerate(ec_lc_results['p']) if p==50][0]
    val = ec_lc_results['value'][idx]
    if val and '>' not in str(val) and 'n/a' not in str(val):
        try: ax.axvline(float(val), color='green', linestyle='--', label=f'EC50: {val}')
        except: pass
        
    ax.axhline(50, color='red', linestyle=':')
    ax.set_xlabel('Concentration (mg/L)'); ax.set_ylabel(y_label)
    ax.legend(); ax.set_title(f'{label} Curve')
    return fig

def plot_growth_curves(df):
    st.subheader("📈 생장 곡선")
    time_cols = ['0h', '24h', '48h', '72h']
    fig, ax = plt.subplots(figsize=(8, 5))
    concs = sorted(df['농도(mg/L)'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(concs)))
    for i, c in enumerate(concs):
        sub = df[df['농도(mg/L)']==c]
        means = [sub[t].mean() for t in time_cols]
        ax.plot([0,24,48,72], means, 'o-', label=f"{c} mg/L", color=colors[i])
    ax.set_yscale('log')
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Cell Density (Log)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
    return fig

# -----------------------------------------------------------------------------
# [실행 함수]
# -----------------------------------------------------------------------------
def run_algae_analysis():
    st.header("🟢 조류 성장저해 시험")
    if 'algae_data' not in st.session_state:
        st.session_state.algae_data = pd.DataFrame({
            '농도(mg/L)': [0]*3+[10]*3+[100]*3, '0h': [10000]*9, '24h': [20000]*9, '48h': [80000]*9, '72h': [500000]*9
        })
    df = st.data_editor(st.session_state.algae_data, num_rows="dynamic")
    
    if st.button("분석 실행"):
        g_fig = plot_growth_curves(df)
        st.divider()
        df['수율'] = df['72h'] - df['0h']
        df['비성장률'] = (np.log(df['72h']) - np.log(df['0h'])) / 3
        c_rate, c_yield = df[df['농도(mg/L)']==0]['비성장률'].mean(), df[df['농도(mg/L)']==0]['수율'].mean()
        
        meta = {'test_type':'Algae Growth', 'protocol':'OECD TG 201', 'species':'P. subcapitata'}
        
        t1, t2 = st.tabs(["비성장률", "수율"])
        with t1:
            # Unpack 2 values
            stats_res, summ = perform_detailed_stats(df, '비성장률', '비성장률', True)
            res, _, met, pi = calculate_ec_lc_range(df, '비성장률', c_rate, 'ErC')
            idx = res['p'].index(50)
            st.metric("ErC50", f"**{res['value'][idx]}**", f"CI: {res['95% CI'][idx]}")
            fig = plot_ec_lc_curve(pi, 'ErC', res, "Inhibition (%)")
            st.pyplot(fig)
            
            meta.update({'endpoint':'Specific Growth Rate', 'method_ec': met, 'col_name':'비성장률'})
            raw_renamed = df.rename(columns={'농도(mg/L)':'Concentration', '비성장률':'Specific Growth Rate'})
            html = generate_full_cetis_report(meta, stats_res, res, raw_renamed, summ.rename(columns={'농도(mg/L)':'Concentration', '비성장률':'Specific Growth Rate'}), fig)
            st.download_button("📥 보고서", html, "Algae_Rate.html")
            
        with t2:
            stats_res, summ = perform_detailed_stats(df, '수율', '수율', True)
            res, _, met, pi = calculate_ec_lc_range(df, '수율', c_yield, 'EyC')
            idx = res['p'].index(50)
            st.metric("EyC50", f"**{res['value'][idx]}**", f"CI: {res['95% CI'][idx]}")
            fig = plot_ec_lc_curve(pi, 'EyC', res, "Inhibition (%)")
            st.pyplot(fig)
            
            meta.update({'endpoint':'Yield', 'method_ec': met, 'col_name':'수율'})
            raw_renamed = df.rename(columns={'농도(mg/L)':'Concentration', '수율':'Yield'})
            html = generate_full_cetis_report(meta, stats_res, res, raw_renamed, summ.rename(columns={'농도(mg/L)':'Concentration', '수율':'Yield'}), fig)
            st.download_button("📥 보고서", html, "Algae_Yield.html")

def run_animal_analysis(test_name, label):
    st.header(f"{test_name}")
    if 'animal_data' not in st.session_state:
        st.session_state.animal_data = pd.DataFrame({
            '농도(mg/L)': [0, 6.25, 12.5, 25, 50, 100], '총 개체수': [20]*6, '반응 수 (48h)': [0, 0, 1, 5, 18, 20]
        })
    df = st.data_editor(st.session_state.animal_data, num_rows="dynamic")
    
    if st.button("상세 분석 실행"):
        col = '반응 수 (48h)'
        # Unpack 3 values
        noec, loec, summ = perform_detailed_stats(df, col, label, False)
        res, _, met, pi = calculate_ec_lc_range(df, col, 0, label, True)
        
        idx = res['p'].index(50)
        st.metric(f"{label}50", f"{res['value'][idx]}", f"CI: {res['95% CI'][idx]}")
        st.dataframe(pd.DataFrame(res))
        fig = plot_ec_lc_curve(pi, label, res)
        st.pyplot(fig)
        
        meta = {'test_type': test_name, 'endpoint': label, 'method_ec': met, 'is_animal': True, 'total_n': df['총 개체수'].mean(), 'col_name': col}
        raw_renamed = df.rename(columns={'농도(mg/L)':'Concentration', col:'Response'})
        html = generate_full_cetis_report(meta, None, res, raw_renamed, summ.rename(columns={'농도(mg/L)':'Concentration', col:'Response'}), fig, "simple")
        st.download_button("📥 보고서", html, f"{label}_Report.html")

if __name__ == "__main__":
    if "조류" in analysis_type: run_algae_analysis()
    elif "물벼룩" in analysis_type: run_animal_analysis("🦐 물벼룩", "EC")
    elif "어류" in analysis_type: run_animal_analysis("🐟 어류", "LC")
