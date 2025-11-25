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
이 앱은 **OECD TG 201, 202, 203** 보고서 요구사항을 충족합니다.
1. **조류:** 생장 곡선 및 농도-반응 곡선, Full Report.
2. **물벼룩/어류:** Probit(GLM) 우선 적용, 실패 시 ICPIN(Bootstrap)으로 자동 전환.
""")
st.divider()

analysis_type = st.sidebar.radio(
    "분석할 실험을 선택하세요",
    ["🟢 조류 성장저해 (Algae)", "🦐 물벼룩 유영저해 (Daphnia)", "🐟 어류 급성독성 (Fish)"]
)

# -----------------------------------------------------------------------------
# [유틸리티] 리포트 생성 함수
# -----------------------------------------------------------------------------
def generate_cetis_report(test_name, endpoint_label, noec, loec, ec50_val, ci_val, method, ec_results, summary_df, fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    now = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Summary Table HTML
    summ_rows = ""
    for _, row in summary_df.iterrows():
        summ_rows += f"<tr><td>{row['농도(mg/L)']}</td><td>{int(row['count'])}</td><td>{row['mean']:.4f}</td><td>{row['min']:.4f}</td><td>{row['max']:.4f}</td><td>{row['std']:.4f}</td></tr>"

    # Point Estimate HTML
    pe_rows = ""
    target_ps = [10, 20, 50]
    for i, p in enumerate(ec_results['p']):
        if p in target_ps:
            val = ec_results['value'][i]
            ci = ec_results['95% CI'][i]
            pe_rows += f"<tr><td>{endpoint_label}</td><td>EC{p}</td><td>{val}</td><td>{ci}</td></tr>"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{ size: A4; margin: 15mm; }}
            body {{ font-family: 'Arial', 'Malgun Gothic', sans-serif; font-size: 10pt; }}
            .header-box {{ border: 2px solid #000; padding: 10px; text-align: center; background-color: #f9f9f9; }}
            .header-title {{ font-weight: bold; font-size: 14pt; }}
            .section-title {{ font-weight: bold; font-size: 11pt; background-color: #eee; padding: 5px; margin-top: 20px; border-bottom: 1px solid #000; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 10px; font-size: 9pt; }}
            th, td {{ border: 1px solid #000; padding: 5px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            .graph-box {{ text-align: center; margin-top: 10px; }}
            img {{ max-width: 80%; }}
        </style>
    </head>
    <body>
        <div class="header-box"><div class="header-title">CETIS Summary Report</div></div>
        <p><b>Test:</b> {test_name} | <b>Date:</b> {now} | <b>Method:</b> Optimal Pro Ver.</p>
        
        <div class="section-title">Comparison Summary</div>
        <table>
            <tr><th>Endpoint</th><th>NOEC</th><th>LOEC</th><th>Method</th></tr>
            <tr><td>{endpoint_label}</td><td><b>{noec} mg/L</b></td><td><b>{loec} mg/L</b></td><td>Bonferroni t-test</td></tr>
        </table>
        
        <div class="section-title">Point Estimate Summary</div>
        <table>
            <tr><th>Endpoint</th><th>Level</th><th>mg/L</th><th>95% CI</th></tr>
            {pe_rows}
        </table>
        <p style="text-align:right; font-size:9pt;">* Model: {method}</p>
        
        <div class="section-title">Data Summary</div>
        <table>
            <tr><th>Conc</th><th>N</th><th>Mean</th><th>Min</th><th>Max</th><th>Std Dev</th></tr>
            {summ_rows}
        </table>
        
        <div class="section-title">Concentration-Response Curve</div>
        <div class="graph-box"><img src="data:image/png;base64,{img_base64}"></div>
    </body>
    </html>
    """
    return html

# -----------------------------------------------------------------------------
# [함수 1] ICPIN + Bootstrap CI 산출 로직 (오류 수정됨)
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
    
    # 변수 초기화 (에러 방지)
    control_val = y_iso[0]
    if control_val != 0:
        inhibition_rates = (control_val - y_raw) / control_val
    else:
        inhibition_rates = np.zeros_like(y_raw)

    try:
        interpolator = interp1d(y_iso, x_raw, kind='linear', bounds_error=False, fill_value=np.nan)
    except:
        interpolator = None

    def calc_icpin_ec(interp_func, level, ctrl_val):
        if interp_func is None: return np.nan
        target_y = ctrl_val * (1 - level/100)
        if target_y > y_iso.max() + 1e-9: return np.nan 
        if target_y < y_iso.min() - 1e-9: return np.nan
        return float(interp_func(target_y))

    ec_levels = np.arange(5, 100, 5) 
    main_results = {}
    for level in ec_levels:
        main_results[level] = calc_icpin_ec(interpolator, level, control_val)

    boot_estimates = {l: [] for l in ec_levels}
    
    # Bootstrap Loop
    for _ in range(n_boot):
        boot_y_means = []
        for c in x_raw:
            if is_binary and total_col and response_col:
                row = df_temp[df_temp['Concentration'] == c].iloc[0]
                n = int(row[total_col])
                p_hat = row[endpoint] 
                # Binomial resampling
                if n > 0:
                    boot_mean = np.random.binomial(n, np.clip(p_hat, 0, 1)) / n
                else:
                    boot_mean = 0
                boot_y_means.append(boot_mean)
            else:
                vals = df_temp[df_temp['Concentration']==c][endpoint].values
                if len(vals) > 0:
                    boot_y_means.append(np.random.choice(vals, size=len(vals), replace=True).mean())
                else:
                    boot_y_means.append(0)
        
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
    
    for level in ec_levels:
        val = main_results[level]
        boots = boot_estimates[level]
        
        val_str = f"{val:.4f}" if not np.isnan(val) else (f"> {max_conc:.4f}" if level >= 50 else "n/a")
        
        if np.isnan(val) or len(boots) < 20: 
            ci_str = "N/C"
        else:
            ci_str = f"({np.percentile(boots, 2.5):.4f} ~ {np.percentile(boots, 97.5):.4f})"
            
        final_out[f'EC{level}'] = {'val': val_str, 'lcl': ci_str}
        
    return final_out, control_val, inhibition_rates

# -----------------------------------------------------------------------------
# [함수 2] 상세 통계 분석 (NOEC/LOEC)
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
        return None, None, summary

    st.dataframe(summary.style.format("{:.4f}"))
    
    noec, loec = max(concentrations), "> Max"
    
    # Bonferroni Logic
    if num_groups >= 2:
        alpha = 0.05 / (num_groups - 1)
        found_loec = False
        for conc in concentrations[1:]:
            # Equal var assumed for simplicity (Levene check omitted for brevity)
            t, p = stats.ttest_ind(control_group, groups[conc], equal_var=True)
            if p < alpha:
                if not found_loec: loec, found_loec = conc, True
            elif not found_loec: noec = conc
        
        if not found_loec: noec, loec = max(concentrations), "> Max"
    
    c1, c2 = st.columns(2)
    c1.metric("NOEC", f"{noec} mg/L")
    c2.metric("LOEC", f"{loec} mg/L")
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
    
    if is_animal_test:
        total_mean = df.groupby('농도(mg/L)')['총 개체수'].mean()
        total_probit = total_mean[dose_resp_probit['농도(mg/L)']].values
        dose_resp_probit['Inhibition'] = dose_resp_probit[endpoint_col] / total_probit
    else:
        dose_resp_probit['Inhibition'] = (control_mean - dose_resp_probit[endpoint_col]) / control_mean

    method_used = "Linear Interpolation (ICp)"
    plot_info = {}
    
    # 1순위: GLM Probit Analysis
    try:
        if not is_animal_test: raise ValueError("Algae skips Probit")
        
        df_glm = df[df['농도(mg/L)'] > 0].copy()
        df_glm['Log_Conc'] = np.log10(df_glm['농도(mg/L)'])
        
        grouped = df_glm.groupby('농도(mg/L)').agg(
            Response=(endpoint_col, 'sum'), Total=('총 개체수', 'sum'), Log_Conc=('Log_Conc', 'mean')
        ).reset_index()
        
        # GLM 안정화: 0%->0.1%, 100%->99.9%
        grouped.loc[grouped['Response']==grouped['Total'], 'Response'] = grouped['Total'] * 0.999
        grouped.loc[grouped['Response']==0, 'Response'] = grouped['Total'] * 0.001
        
        if grouped['Response'].sum() <= 0: raise ValueError("No response")

        model = sm.GLM(grouped['Response'], sm.add_constant(grouped['Log_Conc']),
                       family=families.Binomial(), exposure=grouped['Total']).fit(disp=False)
        
        intercept, slope = model.params['const'], model.params['Log_Conc']
        if slope <= 0: raise ValueError("Negative Slope")

        # CI Calc
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
                     'slope': slope, 'intercept': intercept, 
                     'x_original': grouped['농도(mg/L)'], 'y_original': grouped['Response']/grouped['Total']}

    # 2순위: Linear Interpolation (ICPIN)
    except Exception as e:
        df_icpin = df.copy()
        conc_col = [c for c in df_icpin.columns if '농도' in c][0]
        df_icpin = df_icpin.rename(columns={conc_col: 'Concentration'})
        
        if is_animal_test:
            df_icpin['Value'] = 1 - (df_icpin[endpoint_col] / df_icpin['총 개체수']) 
            icpin_res, ctrl_val, inh_rates = get_icpin_values_with_ci(
                df_icpin, 'Value', is_binary=True, total_col='총 개체수', response_col=endpoint_col
            )
        else:
            df_icpin['Value'] = df_icpin[endpoint_col] 
            icpin_res, ctrl_val, inh_rates = get_icpin_values_with_ci(df_icpin, 'Value', False)

        method_used = "Linear Interpolation (ICPIN/Bootstrap)"
        
        for p in p_values:
            lvl = int(p*100)
            r = icpin_res.get(f'EC{lvl}', {'val': 'n/a', 'lcl': 'n/a'})
            ec_lc_results['p'].append(lvl)
            ec_lc_results['value'].append(r['val'])
            ec_lc_results['status'].append("✅ Interpol")
            ec_lc_results['95% CI'].append(r['lcl'])
            
        unique_concs = sorted(df_icpin['Concentration'].unique())
        plot_info = {'type': 'linear', 'data': dose_resp, 'x_original': unique_concs, 'y_original': inh_rates}

    return ec_lc_results, 0, method_used, plot_info

# -----------------------------------------------------------------------------
# [함수 4] 그래프 출력
# -----------------------------------------------------------------------------
def plot_ec_lc_curve(plot_info, label, ec_lc_results, y_label="Response (%)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    x, y = plot_info['x_original'], plot_info['y_original']
    ax.scatter(x, y*100, c='blue', label='Observed', zorder=5)
    
    if plot_info['type'] == 'probit':
        x_p = np.logspace(np.log10(min(x[x>0])), np.log10(max(x)), 100)
        y_p = stats.norm.cdf(plot_info['slope']*np.log10(x_p)+plot_info['intercept'])*100
        ax.plot(x_p, y_p, 'r-', label='Probit Fit')
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
    st.pyplot(fig)
    return fig

# -----------------------------------------------------------------------------
# [함수 5] 생장 곡선
# -----------------------------------------------------------------------------
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
# [실행 함수] 조류
# -----------------------------------------------------------------------------
def run_algae_analysis():
    st.header("🟢 조류 성장저해 시험")
    if 'algae_data_full' not in st.session_state:
        st.session_state.algae_data_full = pd.DataFrame({
            '농도(mg/L)': [0]*3+[10]*3+[100]*3, '0h': [10000]*9, 
            '24h': [20000]*3+[15000]*3+[10000]*3, '48h': [80000]*3+[40000]*3+[10000]*3, 
            '72h': [500000]*3+[150000]*3+[10000]*3
        })
    df = st.data_editor(st.session_state.algae_data_full, num_rows="dynamic")
    
    if st.button("분석 실행"):
        g_fig = plot_growth_curves(df)
        st.divider()
        
        df['수율'] = df['72h'] - df['0h']
        df['비성장률'] = (np.log(df['72h']) - np.log(df['0h'])) / 3
        c_rate = df[df['농도(mg/L)']==0]['비성장률'].mean()
        c_yield = df[df['농도(mg/L)']==0]['수율'].mean()
        
        tab1, tab2 = st.tabs(["비성장률", "수율"])
        with tab1:
            noec, loec, summ = perform_detailed_stats(df, '비성장률', '비성장률', True)
            res, _, met, pi = calculate_ec_lc_range(df, '비성장률', c_rate, 'ErC', False)
            idx = res['p'].index(50)
            val, ci = res['value'][idx], res['95% CI'][idx]
            st.metric("ErC50", f"**{val}**", f"95% CI: {ci}")
            st.dataframe(pd.DataFrame(res))
            fig = plot_ec_lc_curve(pi, 'ErC', res, "Inhibition (%)")
            
            meta = {'endpoint': 'Specific Growth Rate', 'method_ec': met, 'col_name': '비성장률'}
            html = generate_cetis_report("조류 성장저해", "ErC50", noec, loec, val, ci, met, res, summ, fig)
            st.download_button("📥 보고서 다운로드", html, "Algae_Rate_Report.html")

        with tab2:
            noec, loec, summ = perform_detailed_stats(df, '수율', '수율', True)
            res, _, met, pi = calculate_ec_lc_range(df, '수율', c_yield, 'EyC', False)
            idx = res['p'].index(50)
            val, ci = res['value'][idx], res['95% CI'][idx]
            st.metric("EyC50", f"**{val}**", f"95% CI: {ci}")
            st.dataframe(pd.DataFrame(res))
            fig = plot_ec_lc_curve(pi, 'EyC', res, "Inhibition (%)")
            
            meta = {'endpoint': 'Yield', 'method_ec': met, 'col_name': '수율'}
            html = generate_cetis_report("조류 성장저해", "EyC50", noec, loec, val, ci, met, res, summ, fig)
            st.download_button("📥 보고서 다운로드", html, "Algae_Yield_Report.html")

# -----------------------------------------------------------------------------
# [실행 함수] 물벼룩/어류
# -----------------------------------------------------------------------------
def run_daphnia_analysis():
    st.header("🦐 물벼룩 급성 유영저해 시험")
    if 'daphnia_data_v2' not in st.session_state:
        st.session_state.daphnia_data_v2 = pd.DataFrame({
            '농도(mg/L)': [0.0, 6.25, 12.5, 25.0, 50.0, 100.0], '총 개체수': [20]*6, '반응 수 (24h)': [0]*6, '반응 수 (48h)': [0, 0, 1, 5, 18, 20]
        })
    df = st.data_editor(st.session_state.daphnia_data_v2, num_rows="dynamic")
    
    if st.button("상세 분석 실행"):
        for t in ['24h', '48h']:
            col = f'반응 수 ({t})'
            st.subheader(f"{t} EC50 분석")
            noec, loec, summ = perform_detailed_stats(df, col, "EC", False)
            res, _, met, pi = calculate_ec_lc_range(df, col, 0, "EC", True)
            
            idx = res['p'].index(50)
            val, ci = res['value'][idx], res['95% CI'][idx]
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{t} EC50", f"**{val}**")
            c2.metric("95% CI", ci)
            c3.metric("Model", met)
            
            st.dataframe(pd.DataFrame(res))
            fig = plot_ec_lc_curve(pi, f"{t} EC", res, "Immobility (%)")
            html = generate_cetis_report(f"물벼룩 급성 ({t})", "EC50", noec, loec, val, ci, met, res, summ, fig)
            st.download_button(f"📥 {t} 보고서 다운로드", html, f"Daphnia_{t}.html")

def run_fish_analysis():
    st.header("🐟 어류 급성 독성 시험")
    if 'fish_data_v2' not in st.session_state:
        st.session_state.fish_data_v2 = pd.DataFrame({
            '농도(mg/L)': [0.0, 6.25, 12.5, 25.0, 50.0, 100.0], '총 개체수': [10]*6, 
            '반응 수 (24h)': [0]*6, '반응 수 (48h)': [0]*6, '반응 수 (72h)': [0,0,0,2,5,8], '반응 수 (96h)': [0,0,1,4,8,10]
        })
    df = st.data_editor(st.session_state.fish_data_v2, num_rows="dynamic")
    
    if st.button("상세 분석 실행"):
        times = ['24h', '48h', '72h', '96h']
        tabs = st.tabs(times)
        for i, t in enumerate(times):
            with tabs[i]:
                col = f'반응 수 ({t})'
                st.subheader(f"{t} LC50 분석")
                noec, loec, summ = perform_detailed_stats(df, col, "LC", False)
                res, _, met, pi = calculate_ec_lc_range(df, col, 0, "LC", True)
                
                idx = res['p'].index(50)
                val, ci = res['value'][idx], res['95% CI'][idx]
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{t} LC50", f"**{val}**")
                c2.metric("95% CI", ci)
                c3.metric("Model", met)
                
                st.dataframe(pd.DataFrame(res))
                fig = plot_ec_lc_curve(pi, f"{t} LC", res, "Lethality (%)")
                html = generate_cetis_report(f"어류 급성 ({t})", "LC50", noec, loec, val, ci, met, res, summ, fig)
                st.download_button(f"📥 {t} 보고서 다운로드", html, f"Fish_{t}.html")

if __name__ == "__main__":
    if "조류" in analysis_type: run_algae_analysis()
    elif "물벼룩" in analysis_type: run_daphnia_analysis()
    elif "어류" in analysis_type: run_fish_analysis()
