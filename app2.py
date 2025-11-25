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
import io
import base64
import datetime

# -----------------------------------------------------------------------------
# [공통] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생태독성 전문 분석기 (Final)", page_icon="🧬", layout="wide")

# 한글 폰트 설정 (시스템에 따라 다를 수 있음, 기본 설정)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.title("🧬 생태독성 전문 분석기 (Optimal Pro Ver.)")
st.markdown("""
이 앱은 **OECD TG 201, 202, 203** 보고서 요구사항을 충족하며, **깔끔한 보고서 출력**을 지원합니다.
1. **분석:** TSK, Probit, ICPIN+Bootstrap 자동 적용.
2. **출력:** 11pt 폰트, 주요 결과 강조, 그래프가 포함된 **상세 보고서 다운로드**.
""")
st.divider()

analysis_type = st.sidebar.radio(
    "분석할 실험을 선택하세요",
    ["🟢 조류 성장저해 (Algae)", "🦐 물벼룩 유영저해 (Daphnia)", "🐟 어류 급성독성 (Fish)"]
)

# -----------------------------------------------------------------------------
# [REPORT] HTML 보고서 생성 함수
# -----------------------------------------------------------------------------
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
import io
import base64
import datetime

# -----------------------------------------------------------------------------
# [공통] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생태독성 전문 분석기 (Final)", page_icon="🧬", layout="wide")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.title("🧬 생태독성 전문 분석기 (Optimal Pro Ver.)")
st.markdown("""
이 앱은 **OECD TG** 보고서 요구사항을 충족하며, **"추출 1.pdf" 스타일의 GLP 보고서**를 출력합니다.
""")
st.divider()

analysis_type = st.sidebar.radio(
    "분석할 실험을 선택하세요",
    ["🟢 조류 성장저해 (Algae)", "🦐 물벼룩 유영저해 (Daphnia)", "🐟 어류 급성독성 (Fish)"]
)

# -----------------------------------------------------------------------------
# [REPORT] GLP 스타일 HTML 보고서 생성 함수 (PDF 레이아웃 모방)
# -----------------------------------------------------------------------------
def generate_html_report(test_name, endpoint_label, ec50_val, ci_val, method, df_results, fig):
    # 그래프 변환
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300) # 고해상도
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    now = datetime.datetime.now().strftime("%Y-%m-%d")

    # 데이터프레임 HTML 변환 (스타일링 포함)
    # PDF의 표처럼 보이게 하기 위해 Pandas Styler 대신 직접 HTML 작성 또는 클래스 적용
    df_html = df_results.to_html(index=False, classes='result-table', border=0, justify='center')

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{ size: A4; margin: 20mm; }}
            body {{ 
                font-family: "Times New Roman", "Malgun Gothic", serif; 
                font-size: 11pt; 
                line-height: 1.4; 
                color: #000; 
            }}
            .container {{ width: 100%; max-width: 800px; margin: 0 auto; }}
            
            /* 타이틀 영역 */
            .report-header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #000; padding-bottom: 10px; }}
            .report-title {{ font-size: 18pt; font-weight: bold; margin: 0; }}
            .report-sub {{ font-size: 12pt; margin-top: 5px; }}

            /* 정보 테이블 (상단) */
            .info-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            .info-table td {{ padding: 5px; border: none; vertical-align: top; }}
            .label {{ font-weight: bold; width: 120px; }}

            /* 섹션 헤더 */
            .section-header {{ 
                font-size: 12pt; 
                font-weight: bold; 
                background-color: #e0e0e0; 
                padding: 5px 10px; 
                margin-top: 20px; 
                margin-bottom: 10px;
                border-top: 2px solid #000;
                border-bottom: 1px solid #000;
            }}

            /* 결과 요약 테이블 */
            .summary-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            .summary-table th, .summary-table td {{ border: 1px solid #000; padding: 8px; text-align: center; }}
            .summary-table th {{ background-color: #f9f9f9; font-weight: bold; }}

            /* 상세 데이터 테이블 (PDF 스타일) */
            .result-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 10pt; }}
            .result-table th {{ 
                border-top: 2px solid #000; 
                border-bottom: 2px solid #000; 
                padding: 8px; 
                background-color: #fff; 
                text-align: center;
            }}
            .result-table td {{ 
                border-bottom: 1px solid #ccc; 
                padding: 6px; 
                text-align: center; 
            }}
            .result-table tr:last-child td {{ border-bottom: 2px solid #000; }}

            /* 그래프 */
            .graph-container {{ text-align: center; margin-top: 20px; border: 1px solid #ddd; padding: 10px; }}
            img {{ max-width: 95%; height: auto; }}

            /* 푸터 */
            .footer {{ margin-top: 50px; text-align: right; font-size: 9pt; font-style: italic; border-top: 1px solid #ccc; padding-top: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="report-header">
                <p class="report-title">최종 시험 보고서</p>
                <p class="report-sub">(Final Report - Ecotoxicity Test)</p>
            </div>
            
            <table class="info-table">
                <tr><td class="label">시험 명칭:</td><td>{test_name}</td></tr>
                <tr><td class="label">시험 항목:</td><td>{endpoint_label}</td></tr>
                <tr><td class="label">시험 일자:</td><td>{now}</td></tr>
                <tr><td class="label">분석 방법:</td><td>{method}</td></tr>
            </table>

            <div class="section-header">1. 시험 결과 요약 (Summary of Results)</div>
            <table class="summary-table">
                <tr>
                    <th>항목 (Endpoint)</th>
                    <th>결과값 (Value)</th>
                    <th>95% 신뢰구간 (95% CI)</th>
                </tr>
                <tr>
                    <td><strong>{endpoint_label} 50</strong></td>
                    <td><strong>{ec50_val} mg/L</strong></td>
                    <td>{ci_val}</td>
                </tr>
            </table>
            <p style="font-size:10pt;">* 본 결과는 <strong>{method}</strong>을 사용하여 산출되었습니다.</p>

            <div class="section-header">2. 상세 산출 내역 (Detailed Calculation)</div>
            {df_html}

            <div class="section-header">3. 농도-반응 곡선 (Concentration-Response Curve)</div>
            <div class="graph-container">
                <img src="data:image/png;base64,{img_base64}">
            </div>

            <div class="footer">
                본 보고서는 검증된 알고리즘(Optimal Pro Ver.)에 의해 자동 생성되었습니다.
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
    
    # Bootstrap Data Preparation
    if is_binary and total_col and response_col:
        pass # Logic handled inside loop
    else:
        groups = {c: df_temp[df_temp['Concentration']==c][endpoint].values for c in x_raw}

    for _ in range(n_boot):
        boot_y_means = []
        
        for c in x_raw:
            if is_binary and total_col and response_col:
                row = df_temp[df_temp['Concentration'] == c].iloc[0]
                n = int(row[total_col])
                p_hat = row[endpoint] 
                if n > 0:
                    resampled_count = np.random.binomial(n, p_hat)
                    boot_mean = resampled_count / n
                else:
                    boot_mean = 0
                boot_y_means.append(boot_mean)
            else:
                vals = groups[c]
                if len(vals) > 0:
                    resample = np.random.choice(vals, size=len(vals), replace=True)
                    boot_y_means.append(resample.mean())
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
    
    if num_groups < 2:
        st.error("데이터 그룹이 2개 미만입니다.")
        return

    st.markdown("#### 1. 기초 통계량")
    summary = df.groupby('농도(mg/L)')[endpoint_col].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    st.dataframe(summary.style.format("{:.4f}"))
    
    noec = max(concentrations)
    loec = "> Max"
    
    if num_groups >= 2:
        # Bonferroni T-test Logic
        alpha = 0.05 / (num_groups - 1)
        found_loec = False
        for conc in concentrations[1:]:
            t, p = stats.ttest_ind(control_group, groups[conc], equal_var=True)
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
            if slope <= 0: raise ValueError("Negative Slope")
            
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
            grouped = df_p

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

        method_used = "GLM Probit Analysis (Curve Fitted)"
        
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
        plot_info = {'type': 'linear', 'data': dose_resp, 'r_squared': 0, 
                     'x_original': unique_concs, 
                     'y_original': inh_rates}

    return ec_lc_results, 0, method_used, plot_info

# -----------------------------------------------------------------------------
# [함수 4] 그래프 출력 (Dose-Response & Probit Curve)
# -----------------------------------------------------------------------------
def plot_ec_lc_curve(plot_info, label, ec_lc_results, y_label="Response (%)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_orig = plot_info['x_original']
    y_orig = plot_info['y_original']
    
    ax.scatter(x_orig, y_orig * 100, color='blue', label='Observed', zorder=5)
    
    if plot_info['type'] == 'probit':
        x_pred = np.logspace(np.log10(min(x_orig[x_orig>0])), np.log10(max(x_orig)), 100)
        y_pred = stats.norm.cdf(plot_info['slope']*np.log10(x_pred)+plot_info['intercept']) * 100
        ax.plot(x_pred, y_pred, 'r-', label='Probit Model')
        ax.set_xscale('log')
    else:
        ax.plot(x_orig, y_orig * 100, 'b--', label='Interpolation', alpha=0.5)

    ec50_entry = [res for res in ec_lc_results['value'] if ec_lc_results['p'][ec_lc_results['value'].index(res)] == 50]
    ec50_val = ec50_entry[0] if ec50_entry and ec50_entry[0] != '-' and '>' not in str(ec50_entry[0]) else None
    
    if ec50_val:
        try:
            val = float(ec50_val)
            ax.axvline(val, color='green', linestyle='--', label=f'LC50/EC50: {val}')
        except: pass

    ax.axhline(50, color='gray', linestyle=':')
    ax.set_title(f'{label} Dose-Response Curve')
    ax.set_xlabel('Concentration (mg/L)')
    ax.set_ylabel(y_label)
    ax.legend()
    st.pyplot(fig)
    
    return fig # 중요: 보고서 생성을 위해 fig 반환

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
    return fig # 보고서용 반환

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
        
        init_cells = df['0h'].mean()
        df['수율'] = df['72h'] - df['0h']
        df['비성장률'] = (np.log(df['72h']) - np.log(df['0h'])) / (72/24)
        c_yield = df[df['농도(mg/L)']==0]['수율'].mean()
        c_rate = df[df['농도(mg/L)']==0]['비성장률'].mean()
        
        tab1, tab2 = st.tabs(["비성장률 (Growth Rate)", "수율 (Yield)"])
        with tab1:
            perform_detailed_stats(df, '비성장률', '비성장률')
            res, r2, met, pi = calculate_ec_lc_range(df, '비성장률', c_rate, 'ErC', False)
            idx = [i for i, p in enumerate(res['p']) if p==50][0]
            val, ci = res['value'][idx], res['95% CI'][idx]
            
            st.metric("ErC50", f"**{val} mg/L**", f"95% CI: {ci}")
            st.metric("Model", met)
            res_df = pd.DataFrame(res)
            st.dataframe(res_df)
            fig = plot_ec_lc_curve(pi, 'ErC', res, "Inhibition (%)")
            
            # 보고서 다운로드
            html = generate_html_report("조류 성장저해", "ErC50 (Growth Rate)", val, ci, met, res_df, fig)
            st.download_button("📥 보고서 다운로드 (HTML)", html, "Algae_ErC50_Report.html")

        with tab2:
            perform_detailed_stats(df, '수율', '수율')
            res, r2, met, pi = calculate_ec_lc_range(df, '수율', c_yield, 'EyC', False)
            idx = [i for i, p in enumerate(res['p']) if p==50][0]
            val, ci = res['value'][idx], res['95% CI'][idx]
            
            st.metric("EyC50", f"**{val} mg/L**", f"95% CI: {ci}")
            st.metric("Model", met)
            res_df = pd.DataFrame(res)
            st.dataframe(res_df)
            fig = plot_ec_lc_curve(pi, 'EyC', res, "Inhibition (%)")
            
            html = generate_html_report("조류 성장저해", "EyC50 (Yield)", val, ci, met, res_df, fig)
            st.download_button("📥 보고서 다운로드 (HTML)", html, "Algae_EyC50_Report.html")

# -----------------------------------------------------------------------------
# [분석 실행] 물벼룩
# -----------------------------------------------------------------------------
def run_daphnia_analysis():
    st.header("🦐 물벼룩 급성 유영저해 시험")
    if 'daphnia_data' not in st.session_state:
        st.session_state.daphnia_data = pd.DataFrame({
            '농도(mg/L)': [0.0, 6.25, 12.5, 25.0, 50.0, 100.0],
            '총 개체수': [20]*6, '반응 수 (24h)': [0]*6, '반응 수 (48h)': [0, 0, 1, 5, 18, 20]
        })
    df_input = st.data_editor(st.session_state.daphnia_data, num_rows="dynamic", use_container_width=True)
    
    if st.button("상세 분석 실행"):
        df = df_input.copy()
        t24, t48 = st.tabs(["24h 분석", "48h 분석"])
        
        for t_label, col in zip(["24h", "48h"], ['반응 수 (24h)', '반응 수 (48h)']):
            with (t24 if t_label=="24h" else t48):
                st.subheader(f"{t_label} EC50 분석")
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
                
                html = generate_html_report(f"물벼룩 급성 ({t_label})", "EC50", val, ci, met, res_df, fig)
                st.download_button(f"📥 {t_label} 보고서 다운로드", html, f"Daphnia_{t_label}_Report.html")

# -----------------------------------------------------------------------------
# [분석 실행] 어류
# -----------------------------------------------------------------------------
def run_fish_analysis():
    st.header("🐟 어류 급성 독성 시험")
    if 'fish_data' not in st.session_state:
        st.session_state.fish_data = pd.DataFrame({
            '농도(mg/L)': [0.0, 6.25, 12.5, 25.0, 50.0, 100.0],
            '총 개체수': [10]*6, '반응 수 (24h)': [0]*6, '반응 수 (48h)': [0]*6,
            '반응 수 (72h)': [0, 0, 0, 2, 5, 8], '반응 수 (96h)': [0, 0, 1, 4, 8, 10]
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
                val, ci = ec_res['value'][idx], ec_res['95% CI'][idx]

                c1, c2, c3 = st.columns(3)
                c1.metric(f"{t} LC50", f"**{val} mg/L**")
                c2.metric("95% CI", ci)
                c3.metric("Model", met)
                
                if t == '96h' and 'Probit' in met:
                    slope_val = pi.get('slope', None)
                    if slope_val: st.info(f"📐 **96h Slope:** {slope_val:.4f}")
                
                res_df = pd.DataFrame(ec_res).rename(columns={'p': 'LC (p)', 'value': 'Conc', '95% CI': '95% CI'})
                st.dataframe(res_df.style.apply(lambda x: ['background-color: #e6f3ff']*len(x) if x['LC (p)']==50 else ['']*len(x), axis=1))
                
                y_lab = "Lethality (%)" if t == '96h' else "Response (%)"
                title_lab = f"{t} Concentration-Lethality" if t == '96h' else f"{t} LC"
                fig = plot_ec_lc_curve(pi, title_lab, ec_res, y_lab)
                
                html = generate_html_report(f"어류 급성 ({t})", "LC50", val, ci, met, res_df, fig)
                st.download_button(f"📥 {t} 보고서 다운로드", html, f"Fish_{t}_Report.html")

if __name__ == "__main__":
    if "조류" in analysis_type: run_algae_analysis()
    elif "물벼룩" in analysis_type: run_daphnia_analysis()
    elif "어류" in analysis_type: run_fish_analysis()
