import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.genmod import families
from scipy.stats import norm 
from scipy.interpolate import interp1d 
from scipy.optimize import curve_fit
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
이 앱은 **CETIS Summary Report** 스타일을 완벽하게 재현합니다.
1. **분석:** **Linearized Probit (Robust)** 엔진 탑재.
2. **검증:** **P-value 및 통계 검정 상세 결과** 제공 (NOEC 검증용).
""")

# -----------------------------------------------------------------------------
# [사이드바] 시험 정보 입력
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📝 시험 정보 입력")
    analysis_type = st.radio(
        "분석할 실험을 선택하세요",
        ["🟢 조류 성장저해 (Algae)", "🦐 물벼룩 유영저해 (Daphnia)", "🐟 어류 급성독성 (Fish)"]
    )
    st.divider()
    meta_input = {
        "study_no": st.text_input("Study No.", "GT21-00035"),
        "test_item": st.text_input("Test Item", "Test Substance A"),
        "sponsor": st.text_input("Sponsor", "Korea Environment Corp."),
        "batch_id": st.text_input("Batch ID", "200716P1"),
        "analyst": st.text_input("Analyst", "Analyst Name"),
        "protocol": st.text_input("Protocol", "OECD TG"),
        "start_date": st.date_input("Start Date", datetime.date.today())
    }

# -----------------------------------------------------------------------------
# [유틸리티] 그래프 변환
# -----------------------------------------------------------------------------
def fig_to_base64(fig):
    if fig is None: return ""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_str
    except: return ""

# -----------------------------------------------------------------------------
# [GRAPH] 그래프 함수들
# -----------------------------------------------------------------------------
def plot_raw_response(df, x_col, y_col, y_label):
    fig, ax = plt.subplots(figsize=(6, 4))
    # 데이터 타입 안전 변환
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col])
    
    summary = df.groupby(x_col)[y_col].agg(['mean', 'sem']).reset_index()
    ax.scatter(df[x_col], df[y_col], color='orange', alpha=0.6, label='Observed', zorder=2)
    ax.errorbar(summary[x_col], summary['mean'], yerr=summary['sem'], fmt='-o', color='red', capsize=5, label='Mean ± SE', zorder=3)
    ax.set_xlabel("Concentration (mg/L)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} vs Concentration")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    return fig

def plot_qq_rankits(df, x_col, y_col):
    df_temp = df.copy()
    df_temp[x_col] = pd.to_numeric(df_temp[x_col], errors='coerce')
    df_temp[y_col] = pd.to_numeric(df_temp[y_col], errors='coerce')
    df_temp = df_temp.dropna(subset=[x_col, y_col])
    
    df_temp['Group_Mean'] = df_temp.groupby(x_col)[y_col].transform('mean')
    df_temp['Residuals'] = df_temp[y_col] - df_temp['Group_Mean']
    residuals = df_temp['Residuals'].sort_values().values
    n = len(residuals)
    if n < 2: return None
    rankits = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(rankits, residuals, color='blue', alpha=0.7)
    slope, intercept, r_value, _, _ = stats.linregress(rankits, residuals)
    ax.plot(rankits, slope*rankits + intercept, 'r-', label=f'R²={r_value**2:.3f}')
    ax.set_xlabel("Rankits")
    ax.set_ylabel("Centered Untransformed (Residuals)")
    ax.set_title("Q-Q Plot")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    return fig

# -----------------------------------------------------------------------------
# [REPORT] CETIS 스타일 HTML 보고서 생성 함수
# -----------------------------------------------------------------------------
def generate_full_cetis_report(meta_info, stats_results, ec_results, detail_df, summary_df, dose_resp_fig, raw_resp_fig=None, qq_fig=None, growth_fig=None, report_type="full"):
    img_dr = fig_to_base64(dose_resp_fig)
    img_raw = fig_to_base64(raw_resp_fig) if raw_resp_fig else ""
    img_qq = fig_to_base64(qq_fig) if qq_fig else ""
    img_growth = fig_to_base64(growth_fig) if growth_fig else ""
    
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    term_prefix = "LC" if "Lethality" in meta_info.get('endpoint', '') else "EC"
    
    # 1. Point Estimate Rows
    pe_rows = ""
    target_ps = [5, 10, 20, 50, 80, 90, 95] 
    for i, p in enumerate(ec_results['p']):
        if p in target_ps:
            val = ec_results['value'][i]
            ci = ec_results['95% CI'][i]
            row_style = "background-color: #ffffcc; font-weight: bold;" if p == 50 else ""
            pe_rows += f"<tr style='{row_style}'><td>{meta_info['endpoint']}</td><td>{term_prefix}{p}</td><td>{val}</td><td>{ci}</td><td>{meta_info['method_ec']}</td></tr>"

    # 2. Data Summary Rows
    summ_rows = ""
    conc_col = summary_df.columns[0]
    if 'Concentration' in summary_df.columns: conc_col = 'Concentration'
    elif '농도(mg/L)' in summary_df.columns: conc_col = '농도(mg/L)'
    
    if 0 in summary_df[conc_col].values:
        control_mean = summary_df[summary_df[conc_col]==0]['mean'].values[0]
    else: control_mean = 0 
    
    for _, row in summary_df.iterrows():
        n = int(row['count'])
        m = row['mean']
        s = row['std']
        if pd.isna(s): s = 0
        se = s / np.sqrt(n) if n > 0 else 0
        
        ci_min = m - 1.96 * se
        ci_max = m + 1.96 * se
        cv = (s / m * 100) if m != 0 else 0
        
        if report_type == "simple" and meta_info.get('is_animal', False): 
             total_n = meta_info.get('total_n', 10)
             effect = (m / total_n) * 100
        else:
             effect = ((control_mean - m) / control_mean * 100) if control_mean != 0 else 0

        summ_rows += f"""
        <tr><td>{row[conc_col]}</td><td>{n}</td><td>{m:.4f}</td><td>{ci_min:.4f}</td><td>{ci_max:.4f}</td><td>{row['min']:.4f}</td><td>{row['max']:.4f}</td><td>{se:.4f}</td><td>{cv:.2f}%</td><td>{effect:.2f}%</td></tr>"""

    # 3. Detail Data Rows
    detail_html = ""
    try:
        c_col = detail_df.columns[0]
        v_col = detail_df.columns[1]
        temp_detail = detail_df.copy()
        temp_detail['Rep_Num'] = temp_detail.groupby(c_col).cumcount() + 1
        pivot_df = temp_detail.pivot(index=c_col, columns='Rep_Num', values=v_col)
        detail_header = "<th>Conc-mg/L</th>" + "".join([f"<th>Rep {c}</th>" for c in pivot_df.columns])
        detail_body = ""
        for conc, row in pivot_df.iterrows():
            if meta_info.get('is_animal', False):
                 vals = "".join([f"<td>{int(v)}</td>" for v in row])
            else:
                 vals = "".join([f"<td>{v:.4f}</td>" for v in row])
            detail_body += f"<tr><td>{conc}</td>{vals}</tr>"
        detail_html = f"""<div class="section-title">Detail Data ({meta_info['endpoint']} Values)</div><table><tr>{detail_header}</tr>{detail_body}</table>"""
    except: detail_html = ""
    
    # 4. Comparison Detail Rows (*** NEW ***)
    comp_detail_html = ""
    if 'comp_details' in stats_results and stats_results['comp_details'] is not None:
        comp_rows = ""
        for row in stats_results['comp_details']:
            # row: {'conc', 'stat', 'pval', 'sig', 'alpha'}
            style = "font-weight:bold; color:red;" if row['sig'] == "Significant" else ""
            comp_rows += f"<tr style='{style}'><td>{row['conc']}</td><td>{row['stat']}</td><td>{row['pval']}</td><td>{row['alpha']}</td><td>{row['sig']}</td></tr>"
        
        comp_detail_html = f"""
        <div class="section-title">Comparison Details (Bonferroni t-test)</div>
        <table>
            <tr><th>Concentration</th><th>T-Stat</th><th>P-Value</th><th>Alpha</th><th>Result</th></tr>
            {comp_rows}
        </table>
        """

    comparison_html = ""
    assumption_html = ""
    anova_html = ""

    if report_type == "full" and stats_results:
        comparison_html = f"""
        <div class="section-title">Comparison Summary</div>
        <table><tr><th>Endpoint</th><th>NOEC</th><th>LOEC</th><th>Method</th></tr><tr><td>{meta_info['endpoint']}</td><td><b>{stats_results['noec']} mg/L</b></td><td><b>{stats_results['loec']} mg/L</b></td><td>{stats_results['test_name']}</td></tr></table>"""
        assumption_html = f"""
        <div class="section-title">Test Acceptability & Assumptions</div>
        <table><tr><th>Attribute</th><th>Test</th><th>Statistic</th><th>P-Value</th><th>Decision</th></tr><tr><td>Normality</td><td>Shapiro-Wilk</td><td>{stats_results.get('shapiro_stat',0):.4f}</td><td>{stats_results.get('shapiro_p',1):.4f}</td><td>{stats_results.get('shapiro_res','-')}</td></tr><tr><td>Variances</td><td>Levene's Test</td><td>{stats_results.get('levene_stat',0):.4f}</td><td>{stats_results.get('levene_p',1):.4f}</td><td>{stats_results.get('levene_res','-')}</td></tr></table>"""
        if 'anova_f' in stats_results:
             anova_html = f"""<div class="section-title">Analysis of Variance (ANOVA)</div><table><tr><th>Source</th><th>F-Stat</th><th>P-Value</th></tr><tr><td>Between Groups</td><td>{stats_results['anova_f']:.4f}</td><td>{stats_results['anova_p']:.4f}</td></tr></table>"""

    graphics_html = ""
    if report_type == "full":
        graphics_html = f"""
        <div class="page-break"></div><div class="section-title">Graphics</div>
        <table style="border:none; width:100%;"><tr style="background-color:white;"><td style="width:50%; border:none; vertical-align:top; padding:5px;"><div style="font-weight:bold; margin-bottom:5px; font-size:10pt; text-align:center;">Growth Curves</div><img src="data:image/png;base64,{img_growth}" style="width:100%; border:1px solid #ccc;"></td><td style="width:50%; border:none; vertical-align:top; padding:5px;"><div style="font-weight:bold; margin-bottom:5px; font-size:10pt; text-align:center;">Concentration-Response Curve</div><img src="data:image/png;base64,{img_dr}" style="width:100%; border:1px solid #ccc;"></td></tr><tr style="background-color:white;"><td style="width:50%; border:none; vertical-align:top; padding:5px;"><div style="font-weight:bold; margin-bottom:5px; font-size:10pt; text-align:center;">{meta_info['endpoint']} vs Concentration</div><img src="data:image/png;base64,{img_raw}" style="width:100%; border:1px solid #ccc;"></td><td style="width:50%; border:none; vertical-align:top; padding:5px;"><div style="font-weight:bold; margin-bottom:5px; font-size:10pt; text-align:center;">Q-Q Plot (Rankits)</div><img src="data:image/png;base64,{img_qq}" style="width:100%; border:1px solid #ccc;"></td></tr></table>"""
    else:
        graphics_html = f"""<div class="section-title">Graphics - Concentration Response Curve</div><div class="graph-box"><img src="data:image/png;base64,{img_dr}" style="max-width:60%; border:1px solid #ccc;"></div>"""

    html = f"""
    <!DOCTYPE html><html><head><meta charset="utf-8">
    <style>@page {{ size: A4; margin: 15mm; }} body {{ font-family: 'Arial', 'Malgun Gothic', sans-serif; font-size: 10pt; color: #000; line-height: 1.3; }} .header-box {{ border: 2px solid #000; padding: 10px; margin-bottom: 10px; background-color: #f9f9f9; text-align: center; }} .header-title {{ font-weight: bold; font-size: 16pt; }} .section-title {{ font-weight: bold; font-size: 11pt; background-color: #e6e6e6; padding: 5px; margin-top: 20px; border-bottom: 1px solid #000; }} table {{ width: 100%; border-collapse: collapse; margin-bottom: 10px; font-size: 9pt; }} th {{ border: 1px solid #000; background-color: #f2f2f2; padding: 4px; text-align: center; font-weight: bold; }} td {{ border: 1px solid #000; padding: 4px; text-align: center; }} .info-grid td {{ border: none; text-align: left; padding: 2px 5px; }} .info-label {{ font-weight: bold; width: 120px; background-color: #f9f9f9; }} .graph-box {{ text-align: center; margin-top: 10px; }} img {{ max-width: 95%; border: 1px solid #ccc; }} .page-break {{ page-break-before: always; }}</style>
    </head><body>
        <div class="header-box"><div class="header-title">CETIS Summary Report</div></div>
        <table class="info-grid"><tr><td class="info-label">Study No.:</td><td>{meta_info.get('study_no','-')}</td><td class="info-label">Test Item:</td><td>{meta_info.get('test_item','-')}</td></tr><tr><td class="info-label">Sponsor:</td><td>{meta_info.get('sponsor','-')}</td><td class="info-label">Date:</td><td>{now}</td></tr><tr><td class="info-label">Batch ID:</td><td>{meta_info.get('batch_id','-')}</td><td class="info-label">Protocol:</td><td>{meta_info.get('protocol','-')}</td></tr><tr><td class="info-label">Analyst:</td><td>{meta_info.get('analyst','-')}</td><td class="info-label">Endpoint:</td><td>{meta_info['endpoint']}</td></tr></table>
        {comparison_html}
        <div class="section-title">Point Estimate Summary</div><table><tr><th>Endpoint</th><th>Level</th><th>mg/L</th><th>95% CI</th><th>Method</th></tr>{pe_rows}</table>
        <div class="section-title">Summary of Data</div><table><tr><th>Conc</th><th>N</th><th>Mean</th><th>95% LCL</th><th>95% UCL</th><th>Min</th><th>Max</th><th>Std Err</th><th>CV%</th><th>%Effect</th></tr>{summ_rows}</table>
        {comp_detail_html}
        {detail_html}
        {assumption_html}
        {anova_html}
        {graphics_html}
    </body></html>"""
    return html

# -----------------------------------------------------------------------------
# [함수 1] ICPIN + Bootstrap CI
# -----------------------------------------------------------------------------
def get_icpin_values_with_ci(df_resp, endpoint, is_binary=False, total_col=None, response_col=None, n_boot=1000):
    final_out = {}
    control_val = 0
    inhibition_rates = []
    df_temp = df_resp.copy()
    conc_col_name = df_temp.columns[0]
    df_temp = df_temp.rename(columns={conc_col_name: 'Concentration'})
    raw_means = df_temp.groupby('Concentration')[endpoint].mean()
    x_raw = raw_means.index.values.astype(float)
    y_raw = raw_means.values
    control_val = y_raw[0] if len(y_raw) > 0 else 0
    if control_val != 0: inhibition_rates = (control_val - y_raw) / control_val
    else: inhibition_rates = np.zeros_like(y_raw)
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
    for level in ec_levels: main_results[level] = calc_icpin_ec(interpolator, level, control_val)
    boot_estimates = {l: [] for l in ec_levels}
    groups = {} if (is_binary and total_col) else {c: df_temp[df_temp['Concentration']==c][endpoint].values for c in x_raw}
    for _ in range(n_boot):
        boot_y_means = []
        for c in x_raw:
            if is_binary and total_col and response_col:
                row = df_temp[df_temp['Concentration'] == c].iloc[0]
                n, p_hat = int(row[total_col]), row[endpoint]
                boot_mean = np.random.binomial(n, np.clip(p_hat,0,1)) / n if n > 0 else 0
                boot_y_means.append(boot_mean)
            else:
                vals = groups.get(c, [])
                if len(vals)>0: boot_y_means.append(np.random.choice(vals, size=len(vals), replace=True).mean())
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
    for level in ec_levels:
        val = main_results[level]
        boots = boot_estimates[level]
        val_str = f"{val:.4f}" if not np.isnan(val) else (f"> {max_conc:.4f}" if level >= 50 else "n/a")
        if np.isnan(val) or len(boots) < 20: ci_str = "N/C"
        else: ci_str = f"({np.percentile(boots, 2.5):.4f} ~ {np.percentile(boots, 97.5):.4f})"
        final_out[f'EC{level}'] = {'val': val_str, 'lcl': ci_str}
    return final_out, control_val, inhibition_rates, x_raw

# -----------------------------------------------------------------------------
# [함수 2] 상세 통계 분석 (Comparison Detail 추가)
# -----------------------------------------------------------------------------
def perform_detailed_stats(df, endpoint_col, endpoint_name, return_details=False):
    # Force numeric
    df = df.copy()
    for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    st.markdown(f"### 📊 {endpoint_name} 통계 검정 상세 보고서")
    groups = df.groupby(df.columns[0])[endpoint_col].apply(list)
    concentrations = sorted(groups.keys())
    control_group = groups[concentrations[0]]
    num_groups = len(concentrations)
    summary = df.groupby(df.columns[0])[endpoint_col].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    if num_groups < 2: return None, None, summary, None
    
    st.dataframe(summary.style.format("{:.4f}"))
    stats_details = {}
    comp_details = []
    
    resid = []
    for c in concentrations: resid.extend(np.array(groups[c]) - np.mean(groups[c]))
    s_stat, s_p = stats.shapiro(resid) if len(resid)>3 else (0,1)
    stats_details.update({'shapiro_stat': s_stat, 'shapiro_p': s_p, 'shapiro_res': 'Pass' if s_p>0.01 else 'Fail'})
    l_stat, l_p = stats.levene(*[groups[c] for c in concentrations])
    stats_details.update({'levene_stat': l_stat, 'levene_p': l_p, 'levene_res': 'Pass' if l_p>0.01 else 'Fail'})

    noec, loec = max(concentrations), "> Max"
    
    # Bonferroni Correction
    alpha = 0.05 / (num_groups - 1)
    found_loec = False
    
    for conc in concentrations[1:]:
        # P-value Calculation
        t, p = stats.ttest_ind(control_group, groups[conc], equal_var=(l_p>0.01))
        
        # Safety check for NaN p-value (Zero variance case)
        if np.isnan(p):
            if np.mean(control_group) != np.mean(groups[conc]): p = 0.0 # Different means, 0 var -> Significant
            else: p = 1.0 # Same means, 0 var -> Not Significant
        
        is_sig = p < alpha
        sig_str = "Significant" if is_sig else "Not Significant"
        comp_details.append({'conc': conc, 'stat': f"{t:.4f}", 'pval': f"{p:.4e}", 'sig': sig_str, 'alpha': f"{alpha:.4f}"})
        
        if is_sig:
            if not found_loec:
                loec = conc
                found_loec = True
        elif not found_loec:
            noec = conc # Keep updating NOEC until first LOEC found
            
    if not found_loec: noec = max(concentrations)
    
    f_stat, f_p = stats.f_oneway(*[groups[c] for c in concentrations])
    stats_details.update({'anova_f': f_stat, 'anova_p': f_p, 'noec': noec, 'loec': loec, 'test_name': 'Bonferroni t-test'})
    
    st.write(f"**Bonferroni Alpha:** {alpha:.6f}")
    st.table(pd.DataFrame(comp_details))
    
    c1, c2 = st.columns(2)
    c1.metric("NOEC", f"{noec}")
    c2.metric("LOEC", f"{loec}")
    st.divider()
    
    if return_details: return stats_details, summary, comp_details
    return noec, loec, summary, comp_details

# -----------------------------------------------------------------------------
# [함수 3] ECp/LCp 산출
# -----------------------------------------------------------------------------
def calculate_ec_lc_range(df, endpoint_col, control_mean, label, is_animal_test=False):
    df = df.copy()
    for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    if is_animal_test:
        dose_resp = df.groupby(df.columns[0]).agg({endpoint_col: 'sum', '총 개체수': 'sum'}).reset_index()
        dose_resp.columns = ['농도(mg/L)', endpoint_col, '총 개체수']
        dose_resp['Inhibition'] = dose_resp[endpoint_col] / dose_resp['총 개체수']
    else:
        dose_resp = df.groupby(df.columns[0])[endpoint_col].mean().reset_index()
        dose_resp.columns = ['농도(mg/L)', endpoint_col]
        dose_resp['Inhibition'] = (control_mean - dose_resp[endpoint_col]) / control_mean

    max_conc = dose_resp['농도(mg/L)'].max()
    p_values = np.arange(5, 100, 5) / 100
    ec_res = {'p': [], 'value': [], 'status': [], '95% CI': []}
    method_used = "Linear Interpolation (ICp)"
    plot_info = {}

    try:
        if not is_animal_test: raise Exception("Algae: Force ICPIN")
        
        df_reg = dose_resp[dose_resp['농도(mg/L)'] > 0].copy()
        if len(df_reg) < 2: raise ValueError("Not enough points")
        
        df_reg['Prop'] = np.clip(df_reg['Inhibition'], 0.001, 0.999)
        x_data = np.log10(df_reg['농도(mg/L)'].values)
        y_data = norm.ppf(df_reg['Prop'].values)
        
        slope, intercept, r_val, p_val, std_err = stats.linregress(x_data, y_data)
        if slope <= 0: slope = 0.001 
            
        log_ec50 = -intercept / slope
        se_approx = std_err / slope if slope != 0 else 0.1
        lcl_50 = 10**(log_ec50 - 1.96 * se_approx)
        ucl_50 = 10**(log_ec50 + 1.96 * se_approx)
        ci_50 = f"({lcl_50:.4f} ~ {ucl_50:.4f})"

        for p in p_values:
            z = norm.ppf(p)
            log_ecp = (z - intercept) / slope
            ecp = 10**log_ecp
            val_s = f"{ecp:.4f}" if 0 < ecp < max_conc * 1000 else "> Max"
            ec_res['p'].append(int(p*100))
            ec_res['value'].append(val_s)
            ec_res['status'].append("✅ Probit")
            ec_res['95% CI'].append(ci_50 if int(p*100)==50 else "N/A")
            
        method_used = "Linearized Probit Regression"
        x_smooth = np.linspace(min(x_data), max(x_data), 100)
        y_smooth = norm.cdf(slope * x_smooth + intercept)
        y_obs = dose_resp[dose_resp['농도(mg/L)']>0]['Inhibition'].values
        x_obs_plot = dose_resp[dose_resp['농도(mg/L)']>0]['농도(mg/L)'].values
        plot_info = {'type':'probit', 'x': x_smooth, 'y': y_smooth, 'slope': slope, 'intercept': -slope * log_ec50, 'x_original': x_obs_plot, 'y_original': y_obs}
        
    except Exception as e:
        ec_res = {'p': [], 'value': [], 'status': [], '95% CI': []}
        df_icpin = df.copy().rename(columns={df.columns[0]:'Concentration'})
        if is_animal_test:
            df_icpin['Value'] = 1 - (df_icpin[endpoint_col] / df_icpin['총 개체수'])
            icpin_res, _, inh, x_vals = get_icpin_values_with_ci(df_icpin, 'Value', True, '총 개체수', endpoint_col)
        else:
            df_icpin['Value'] = df_icpin[endpoint_col]
            icpin_res, _, inh, x_vals = get_icpin_values_with_ci(df_icpin, 'Value', False)
            
        method_used = "Linear Interpolation (ICPIN/Bootstrap)"
        for p in p_values:
            lvl = int(p*100)
            r = icpin_res.get(f'EC{lvl}', {'val':'n/a', 'lcl':'n/a'})
            ec_res['p'].append(lvl)
            ec_res['value'].append(r['val'])
            ec_res['status'].append("✅ Interpol")
            ec_res['95% CI'].append(r['lcl'])
        plot_info = {'type':'linear', 'x_original': x_vals, 'y_original': inh}

    return ec_res, 0, method_used, plot_info

def plot_ec_lc_curve(plot_info, label, ec_lc_results, y_label="Response (%)"):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = plot_info['x_original']
    y = plot_info['y_original']
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    ax.scatter(x, y*100, c='blue', label='Observed', zorder=5)
    if plot_info['type'] == 'probit':
        x_fit = 10**plot_info['x']
        y_fit = plot_info['y'] * 100 
        ax.plot(x_fit, y_fit, 'r-', label='Probit Fit')
        ax.set_xscale('log')
    else:
        ax.plot(x, y*100, 'b--', label='Interpolation', alpha=0.5)
    
    ec50_val = [ec_lc_results['value'][i] for i, p in enumerate(ec_lc_results['p']) if p==50][0]
    if ec50_val and '>' not in str(ec50_val) and 'N' not in str(ec50_val):
        try: ax.axvline(float(ec50_val), color='green', linestyle='--', label=f'EC50: {ec50_val}')
        except: pass
    ax.axhline(50, color='red', linestyle=':')
    ax.set_xlabel('Concentration (mg/L)'); ax.set_ylabel(y_label)
    ax.legend(); ax.set_title(f'{label} Curve')
    return fig

def plot_growth_curves(df):
    st.subheader("📈 생장 곡선")
    time_cols = ['0h', '24h', '48h', '72h']
    fig, ax = plt.subplots(figsize=(6, 4))
    concs = sorted(df[df.columns[0]].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(concs)))
    for i, c in enumerate(concs):
        sub = df[df[df.columns[0]]==c]
        means = [sub[t].mean() for t in time_cols]
        ax.plot([0,24,48,72], means, 'o-', label=f"{c} mg/L", color=colors[i])
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Cell Density (Cells/mL)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig

# -----------------------------------------------------------------------------
# [실행 함수]
# -----------------------------------------------------------------------------
def run_algae_analysis():
    st.header("🟢 조류 성장저해 시험")
    if 'algae_data_v18' not in st.session_state:
        st.session_state.algae_data_v18 = pd.DataFrame({'농도(mg/L)':[0]*3+[10]*3+[100]*3, '0h':[10000]*9, '24h':[20000]*9, '48h':[80000]*9, '72h':[500000]*9})
    df = st.data_editor(st.session_state.algae_data_v18, num_rows="dynamic")
    if st.button("분석 실행"):
        g_fig = plot_growth_curves(df)
        st.pyplot(g_fig)
        st.divider()
        
        # Calc Rates/Yields
        for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        t0 = df['0h'].replace(0, 1) # Safe log
        t72 = df['72h'].replace(0, 1)
        df['수율'] = t72 - t0
        df['비성장률'] = (np.log(t72) - np.log(t0)) / 3
        
        c_rate = df[df[df.columns[0]]==0]['비성장률'].mean()
        c_yield = df[df[df.columns[0]]==0]['수율'].mean()
        
        meta = meta_input.copy(); meta.update({'test_type': 'Growth Inhibition', 'species': 'P. subcapitata'})
        
        tab1, tab2 = st.tabs(["비성장률", "수율"])
        with tab1:
            stats_res, summ, comps = perform_detailed_stats(df, '비성장률', '비성장률', True)
            # Update stats_res with detailed comparison info
            stats_res['comp_details'] = comps
            
            res, _, met, pi = calculate_ec_lc_range(df, '비성장률', c_rate, 'ErC', False)
            idx = res['p'].index(50)
            st.metric("ErC50", f"**{res['value'][idx]}**", f"CI: {res['95% CI'][idx]}")
            st.dataframe(pd.DataFrame(res))
            fig = plot_ec_lc_curve(pi, 'ErC', res, "Inhibition (%)")
            st.pyplot(fig)
            
            raw_fig = plot_raw_response(df, df.columns[0], '비성장률', 'Rate (1/d)')
            qq_fig = plot_qq_rankits(df, df.columns[0], '비성장률')
            meta.update({'endpoint':'Specific Growth Rate', 'method_ec': met, 'col_name':'비성장률'})
            df_detail = df[[df.columns[0], '비성장률']].rename(columns={df.columns[0]:'Concentration', '비성장률':'Specific Growth Rate'})
            html = generate_full_cetis_report(meta, stats_res, res, df_detail, summ.rename(columns={summ.columns[0]:'Concentration', '비성장률':'Specific Growth Rate'}), fig, raw_fig, qq_fig, g_fig, "full")
            st.download_button("📥 Report", html, "Algae_Rate.html")
            
        with tab2:
            stats_res, summ, comps = perform_detailed_stats(df, '수율', '수율', True)
            stats_res['comp_details'] = comps
            
            res, _, met, pi = calculate_ec_lc_range(df, '수율', c_yield, 'EyC', False)
            idx = res['p'].index(50)
            st.metric("EyC50", f"**{res['value'][idx]}**", f"CI: {res['95% CI'][idx]}")
            st.dataframe(pd.DataFrame(res))
            fig = plot_ec_lc_curve(pi, 'EyC', res, "Inhibition (%)")
            st.pyplot(fig)
            
            raw_fig = plot_raw_response(df, df.columns[0], '수율', 'Yield')
            qq_fig = plot_qq_rankits(df, df.columns[0], '수율')
            meta.update({'endpoint':'Yield', 'method_ec': met, 'col_name':'수율'})
            df_detail = df[[df.columns[0], '수율']].rename(columns={df.columns[0]:'Concentration', '수율':'Yield'})
            html = generate_full_cetis_report(meta, stats_res, res, df_detail, summ.rename(columns={summ.columns[0]:'Concentration', '수율':'Yield'}), fig, raw_fig, qq_fig, g_fig, "full")
            st.download_button("📥 Report", html, "Algae_Yield.html")

def run_daphnia_analysis():
    st.header("🦐 물벼룩")
    if 'd_data_v17' not in st.session_state: st.session_state.d_data_v17 = pd.DataFrame({'농도(mg/L)':[0,6.25,12.5,25,50,100], '총 개체수':[20]*6, '반응 수 (24h)':[0,0,0,0,0,0], '반응 수 (48h)':[0,0,1,5,18,20]})
    df = st.data_editor(st.session_state.d_data_v17, num_rows="dynamic")
    if st.button("분석"):
        t24, t48 = st.tabs(["24h", "48h"])
        for t, col in zip(['24h','48h'], ['반응 수 (24h)','반응 수 (48h)']):
            with (t24 if t=='24h' else t48):
                noec, loec, summ, comps = perform_detailed_stats(df, col, "EC", True)
                stats_res = {'noec':noec, 'loec':loec, 'test_name':'Bonferroni', 'comp_details':comps}
                
                res, _, met, pi = calculate_ec_lc_range(df, col, 0, "EC", True)
                idx = res['p'].index(50)
                st.metric(f"EC50", f"**{res['value'][idx]}**", f"CI: {res['95% CI'][idx]}")
                st.dataframe(pd.DataFrame(res))
                fig = plot_ec_lc_curve(pi, f"{t} EC", res, "Immobility (%)")
                st.pyplot(fig)
                meta = meta_input.copy(); meta.update({'test_type':'Daphnia', 'endpoint':'Immobility', 'method_ec':met, 'is_animal':True, 'total_n':df['총 개체수'].mean(), 'col_name':col})
                df_detail = df[[df.columns[0], col]].rename(columns={df.columns[0]:'Concentration', col: 'Response'})
                html = generate_full_cetis_report(meta, stats_res, res, df_detail, summ.rename(columns={summ.columns[0]:'Concentration', col:'Response'}), fig, None, None, None, "simple")
                st.download_button(f"📥 {t} Report", html, f"Daphnia_{t}.html", key=f"d{t}")

def run_fish_analysis():
    st.header("🐟 어류")
    if 'f_data_v17' not in st.session_state: st.session_state.f_data_v17 = pd.DataFrame({'농도(mg/L)':[0,6.25,12.5,25,50,100], '총 개체수':[10]*6, '반응 수 (24h)':[0]*6, '반응 수 (48h)':[0]*6, '반응 수 (72h)':[0,0,0,2,5,8], '반응 수 (96h)':[0,0,1,4,8,10]})
    df = st.data_editor(st.session_state.f_data_v17, num_rows="dynamic")
    if st.button("분석"):
        tabs = st.tabs(['24h','48h','72h','96h'])
        times = ['24h','48h','72h','96h']
        for i, t in enumerate(times):
            with tabs[i]:
                col = f'반응 수 ({t})'
                noec, loec, summ, comps = perform_detailed_stats(df, col, "LC", True)
                stats_res = {'noec':noec, 'loec':loec, 'test_name':'Bonferroni', 'comp_details':comps}
                
                res, _, met, pi = calculate_ec_lc_range(df, col, 0, "LC", True)
                idx = res['p'].index(50)
                st.metric(f"LC50", f"**{res['value'][idx]}**", f"CI: {res['95% CI'][idx]}")
                st.dataframe(pd.DataFrame(res))
                fig = plot_ec_lc_curve(pi, f"{t} LC", res, "Lethality (%)")
                st.pyplot(fig)
                meta = meta_input.copy(); meta.update({'test_type':'Fish', 'endpoint':'Lethality', 'method_ec':met, 'is_animal':True, 'total_n':df['총 개체수'].mean(), 'col_name':col})
                df_detail = df[[df.columns[0], col]].rename(columns={df.columns[0]:'Concentration', col: 'Response'})
                html = generate_full_cetis_report(meta, stats_res, res, df_detail, summ.rename(columns={summ.columns[0]:'Concentration', col:'Response'}), fig, None, None, None, "simple")
                st.download_button(f"📥 {t} Report", html, f"Fish_{t}.html", key=f"f{t}")

if __name__ == "__main__":
    if "조류" in analysis_type: run_algae_analysis()
    elif "물벼룩" in analysis_type: run_daphnia_analysis()
    elif "어류" in analysis_type: run_fish_analysis()
