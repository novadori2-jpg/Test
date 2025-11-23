import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# [공통] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생태독성 전문 분석기 (Final)", page_icon="🧬", layout="wide")

st.title("🧬 생태독성 전문 분석기 (Detailed Pro Ver.)")
st.markdown("""
이 앱은 **CETIS/ToxCalc 수준의 알고리즘**을 적용하되, **모든 통계적 검정 과정을 투명하게 공개**합니다.
1. **통계 검정:** 기초통계 -> 정규성 -> 등분산성 -> (모수/비모수 자동선택) -> NOEC/LOEC 도출
2. **독성값:** Probit 우선 적용, 적합도 미달 시 선형보간법 자동 전환
""")
st.divider()

analysis_type = st.sidebar.radio(
    "분석할 실험을 선택하세요",
    ["🟢 조류 성장저해 (Algae)", "🦐 물벼룩 유영저해 (Daphnia)", "🐟 어류 급성독성 (Fish)"]
)

# -----------------------------------------------------------------------------
# [핵심 로직 1] 상세 통계 분석 및 가설 검정 (NOEC/LOEC)
# -----------------------------------------------------------------------------
def perform_detailed_stats(df, endpoint_col, endpoint_name):
    """
    상세 통계량을 출력하고, 정규성/등분산성 결과에 따라 
    적절한 검정(ANOVA vs Kruskal)을 수행하여 NOEC/LOEC를 찾습니다.
    """
    st.markdown(f"### 📊 {endpoint_name} 통계 검정 상세 보고서")

    # 데이터 그룹화
    groups = df.groupby('농도(mg/L)')[endpoint_col].apply(list)
    concentrations = sorted(groups.keys())
    control_group = groups[0]

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
            # p < 0.01 이면 정규성 위배 (엄격한 기준)
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
        st.error("데이터 그룹이 충분하지 않습니다.")
        return

    l_stat, l_p = stats.levene(*data_list)
    is_homogeneous = l_p > 0.05
    
    st.write(f"- Statistic: {l_stat:.4f}")
    st.write(f"- P-value: **{l_p:.4f}**")
    st.info(f"판정: **{'✅ 등분산 만족' if is_homogeneous else '❌ 이분산 (등분산 위배)'}**")

    # 4. 가설 검정 (NOEC/LOEC)
    st.markdown("#### 4. 유의성 검정 및 NOEC/LOEC 도출")
    
    noec = 0
    loec = None
    comparisons = []
    
    # [Case A] 정규성 위배 -> 비모수 검정
    if not is_normal:
        st.warning("👉 정규성 가정에 위배되므로 **'비모수 검정(Non-Parametric Analysis)'**을 수행합니다.")
        st.markdown("**검정 방법: Kruskal-Wallis Rank Sum Test**")
        
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
                comparisons.append({
                    '비교 농도': conc, 'Method': 'Mann-Whitney', 'P-value': f"{u_p:.4f}", 
                    'Significance': '🚨 유의차 있음' if is_sig else '✅ 차이 없음'
                })
                if is_sig and loec is None:
                    loec = conc
                if not is_sig:
                    noec = conc
        else:
            st.info("그룹 간 통계적으로 유의한 차이가 없습니다.")
            noec = max(concentrations)

    # [Case B] 정규성 만족 -> 모수 검정
    else:
        st.success("👉 정규성 가정을 만족하므로 **'모수 검정(Parametric Analysis)'**을 수행합니다.")
        
        if is_homogeneous:
            st.markdown("**검정 방법: One-way ANOVA (Equal Variance)**")
        else:
            st.markdown("**검정 방법: One-way ANOVA (Welch's correction recommended)**")
            
        f_stat, f_p = stats.f_oneway(*data_list) 
        st.write(f"- ANOVA P-value: **{f_p:.4f}**")
        
        if f_p < 0.05:
            st.write("👉 그룹 간 차이가 유의함. 사후 검정(**Bonferroni t-test**)을 수행합니다.")
            alpha = 0.05 / (len(concentrations) - 1)
            
            for conc in concentrations:
                if conc == 0:
                    continue
                
                t_stat, t_p = stats.ttest_ind(control_group, groups[conc], equal_var=is_homogeneous)
                
                is_sig = t_p < alpha
                method_str = "t-test" if is_homogeneous else "Welch's t-test"
                
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
# [핵심 로직 2] EC50/LC50 산출 (Probit -> Interpolation Fallback)
# -----------------------------------------------------------------------------
def calculate_point_estimate(df, endpoint_col, control_mean, label):
    dose_resp = df.groupby('농도(mg/L)')[endpoint_col].mean().reset_index()
    dose_resp = dose_resp[dose_resp['농도(mg/L)'] > 0].copy() 

    if '반응 수' in df.columns:
        total = df.groupby('농도(mg/L)')['총 개체수'].mean()[dose_resp['농도(mg/L)']].values
        dose_resp['Inhibition'] = dose_resp[endpoint_col] / total
    else:
        dose_resp['Inhibition'] = (control_mean - dose_resp[endpoint_col]) / control_mean

    method_used = "Probit Analysis"
    ec50_val = None
    r_squared = 0
    plot_info = {}
    
    # 1차 시도: Probit
    try:
        dose_resp['Inhibition_adj'] = dose_resp['Inhibition'].clip(0.001, 0.999)
        dose_resp['Probit'] = stats.norm.ppf(dose_resp['Inhibition_adj'])
        dose_resp['Log_Conc'] = np.log10(dose_resp['농도(mg/L)'])

        slope, intercept, r_val, p_val, std_err = stats.linregress(dose_resp['Log_Conc'], dose_resp['Probit'])
        r_squared = r_val ** 2
        
        if r_squared < 0.6 or slope <= 0:
            raise ValueError("Low Fit")

        log_ec50 = -intercept / slope
        ec50_val = 10 ** log_ec50
        
        plot_info = {
            'type': 'probit', 'x': dose_resp['Log_Conc'], 'y': dose_resp['Probit'], 
            'slope': slope, 'intercept': intercept, 'ec50': log_ec50
        }

    # 2차 시도: Linear Interpolation
    except Exception:
        method_used = "Linear Interpolation (ICp)"
        dose_resp = dose_resp.sort_values('Inhibition')
        
        lower = dose_resp[dose_resp['Inhibition'] <= 0.5].max()
        upper = dose_resp[dose_resp['Inhibition'] >= 0.5].min()
        
        if pd.isna(lower['Inhibition']) or pd.isna(upper['Inhibition']):
            ec50_val = None
        else:
            x1, y1 = lower['농도(mg/L)'], lower['Inhibition']
            x2, y2 = upper['농도(mg/L)'], upper['Inhibition']
            if y1 == y2:
                ec50_val = x1
            else:
                ec50_val = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1)
        
        plot_info = {'type': 'linear', 'data': dose_resp, 'ec50': ec50_val}

    return ec50_val, r_squared, method_used, plot_info


# -----------------------------------------------------------------------------
# [분석 실행 함수] 조류 (Algae)
# -----------------------------------------------------------------------------
def run_algae_analysis():
    st.header("🟢 조류 성장저해 시험 (OECD TG 201)")
    
    with st.expander("⚙️ 실험 조건 설정", expanded=True):
        c1, c2 = st.columns(2)
        init_cells = c1.number_input("초기 세포수", value=10000)
        duration = c2.number_input("배양 시간 (h)", value=72)

    if 'algae_data_final' not in st.session_state:
        st.session_state.algae_data_final = pd.DataFrame({
            '농도(mg/L)': [0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 32.0, 32.0, 32.0, 100.0, 100.0, 100.0],
            '최종 세포수 (cells/mL)': [1000000, 1050000, 980000, 900000, 880000, 910000, 500000, 480000, 520000, 150000, 140000, 160000]
        })
    
    df_input = st.data_editor(
        st.session_state.algae_data_final, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "농도(mg/L)": st.column_config.NumberColumn("농도(mg/L)", format="%.2f"),
            "최종 세포수 (cells/mL)": st.column_config.NumberColumn("최종 세포수", format="%d")
        }
    )
    
    if st.button("상세 분석 실행"):
        df = df_input.copy()
        # 파생변수 계산
        df['수율'] = df['최종 세포수 (cells/mL)'] - init_cells
        df['비성장률'] = (np.log(df['최종 세포수 (cells/mL)']) - np.log(init_cells)) / (duration/24)
        
        # ---------------------------------------------------------
        # [복구됨] 생물량 및 성장률 분포 그래프 (Boxplot)
        # ---------------------------------------------------------
        st.divider()
        st.subheader("📊 데이터 분포 시각화 (Boxplot)")
        
        fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        concs = sorted(df['농도(mg/L)'].unique())
        yield_data = [df[df['농도(mg/L)'] == c]['수율'] for c in concs]
        rate_data = [df[df['농도(mg/L)'] == c]['비성장률'] for c in concs]
        
        # 수율 그래프
        ax1.boxplot(yield_data, labels=concs, patch_artist=True, boxprops=dict(facecolor='#D1E8E2'))
        ax1.set_title('Yield (Biomass)')
        ax1.set_xlabel('Concentration (mg/L)')
        ax1.set_ylabel('Yield (Cell Increase)')
        ax1.grid(axis='y', linestyle=':', alpha=0.7)

        # 비성장률 그래프
        ax2.boxplot(rate_data, labels=concs, patch_artist=True, boxprops=dict(facecolor='#F2D7D5'))
        ax2.set_title('Specific Growth Rate')
        ax2.set_xlabel('Concentration (mg/L)')
        ax2.set_ylabel('Growth Rate (1/day)')
        ax2.grid(axis='y', linestyle=':', alpha=0.7)

        st.pyplot(fig_dist)
        st.divider()
        
        # 탭 구성 (상세 통계 및 EC50)
        tab1, tab2 = st.tabs(["📈 비성장률(Rate) 분석", "📉 수율(Yield) 분석"])
        
        def show_results(target_col, name, ec_label):
            # 1. 상세 통계 (NOEC/LOEC)
            perform_detailed_stats(df, target_col, name)
            
            # 2. EC50 산출
            control_mean = df[df['농도(mg/L)']==0][target_col].mean()
            ec50, r2, method, plot_info = calculate_point_estimate(df, target_col, control_mean, ec_label)
            
            st.markdown(f"#### 5. {ec_label} 산출 결과")
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric(f"{ec_label}", f"{ec50:.4f} mg/L" if ec50 else "산출 불가")
            cm2.metric("적용 모델", method)
            cm3.metric("R²", f"{r2:.4f}" if r2 > 0 else "-")
            
            fig, ax = plt.subplots(figsize=(6, 4))
            if plot_info['type'] == 'probit':
                x = plot_info['x']
                slope = plot_info['slope']
                intercept = plot_info['intercept']
                x_line = np.linspace(min(x), max(x), 100)
                
                ax.scatter(x, plot_info['y'], label='Data')
                ax.plot(x_line, slope*x_line + intercept, color='red', label='Probit Fit')
                ax.set_xlabel('Log Concentration')
                ax.set_ylabel('Probit (Inhibition)')
                if plot_info['ec50']:
                    ax.axvline(plot_info['ec50'], color='green', linestyle='--', label='50% Effect')
            else:
                d = plot_info['data']
                ax.plot(d['농도(mg/L)'], d['Inhibition'], marker='o', label='Linear Interp')
                ax.axhline(0.5, color='red', linestyle='--', label='50% Cutoff')
                if plot_info['ec50']:
                    ax.axvline(plot_info['ec50'], color='green', linestyle='--')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with tab1:
            show_results('비성장률', '비성장률', 'ErC50')
        with tab2:
            show_results('수율', '수율', 'EyC50')

# -----------------------------------------------------------------------------
# [분석 실행 함수] 어류/물벼룩
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
    
    if st.button("분석 실행"):
        df = df_input.copy()
        
        st.subheader(f"📊 {label} 산출 결과")
        ec50, r2, method, plot_info = calculate_point_estimate(df, '반응 수', 0, label)
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{label}", f"{ec50:.4f} mg/L" if ec50 else "산출 불가")
        c2.metric("계산 방식", method)
        c3.metric("R²", f"{r2:.4f}" if r2 > 0 else "-")
        
        fig, ax = plt.subplots()
        if plot_info['type'] == 'probit':
            ax.scatter(plot_info['x'], plot_info['y'], label='Data')
            x_line = np.linspace(min(plot_info['x']), max(plot_info['x']), 100)
            ax.plot(x_line, plot_info['slope']*x_line + plot_info['intercept'], color='red')
            ax.set_xlabel('Log Concentration')
            ax.set_ylabel('Probit')
        else:
            d = plot_info['data']
            ax.plot(d['농도(mg/L)'], d['Inhibition'], marker='o')
            ax.set_xlabel('Concentration')
            ax.set_ylabel('Response Rate')
        
        ax.legend()
        st.pyplot(fig)


# -----------------------------------------------------------------------------
# 메인 실행
# -----------------------------------------------------------------------------
if "조류" in analysis_type:
    run_algae_analysis()
elif "물벼룩" in analysis_type:
    run_animal_analysis("🦐 물벼룩 급성 유영저해", "EC50")
elif "어류" in analysis_type:
    run_animal_analysis("🐟 어류 급성 독성", "LC50")
    with col2:
        st.write("#### 등분산성 & 분석 방법 선정")
        st.write(f"- Levene P-value: **{l_p:.4f}** ({'등분산' if is_homogeneous else '이분산'})")
        if not is_normal:
            st.warning("👉 **비모수 검정 (Kruskal-Wallis)** 채택")
            test_type = "non-param"
        else:
            st.success("👉 **모수 검정 (ANOVA)** 채택")
            test_type = "param"

    # (4) 가설 검정 및 사후 검정
    st.write("#### 유의성 검정 결과 (Control vs Treatment)")
    comparisons = []
    noec, loec = max(concentrations), None 

    alpha = 0.05 / (len(concentrations) - 1) if len(concentrations) > 1 else 0.05

    for conc in concentrations:
        if conc == 0: continue
        
        is_sig = False
        p_val = 1.0
        method = ""

        if test_type == "non-param":
            u, p_val = stats.mannwhitneyu(control_group, groups[conc], alternative='two-sided')
            method = "Mann-Whitney"
        else:
            t, p_val = stats.ttest_ind(control_group, groups[conc], equal_var=is_homogeneous)
            method = "Welch's t-test" if not is_homogeneous else "t-test"

        is_sig = p_val < alpha
        
        comparisons.append({
            '비교 농도': conc, 
            'Method': method, 
            'P-value': f"{p_val:.4f}", 
            'Significance': '🚨 유의차 있음 (LOEC 후보)' if is_sig else '✅ 차이 없음'
        })

        if is_sig:
            if loec is None: loec = conc 
        else:
            if loec is None: noec = conc

    st.dataframe(pd.DataFrame(comparisons))
    st.info(f"📍 **결론: NOEC = {noec} mg/L, LOEC = {loec if loec else '> ' + str(max(concentrations))} mg/L**")

# -----------------------------------------------------------------------------
# [모듈 3] 용량-반응 곡선 및 ECx/LCx 전구간 산출 (Hill Equation)
# -----------------------------------------------------------------------------
def hill_equation(x, top, bottom, ec50, hill_slope):
    return bottom + (top - bottom) / (1 + (x / ec50)**(-hill_slope))

def inverse_hill(y, top, bottom, ec50, hill_slope):
    if y >= top: return np.inf
    if y <= bottom: return 0
    return ec50 * (( (top - bottom) / (y - bottom) ) - 1)**(1 / -hill_slope)

def calculate_dose_response(df, endpoint_col):
    st.markdown("### 📈 2. 농도-반응 곡선 및 ECx/LCx 산출")
    
    x_data = df['농도(mg/L)'].values
    y_data = df[endpoint_col].values

    # 초기 추정값 (Top=100, Bottom=0, EC50=Median, Slope=2)
    p0 = [100, 0, np.median(x_data[x_data > 0]), 2]
    bounds = ([90, -10, 0.0001, 0.1], [110, 10, np.inf, 20])

    try:
        popt, pcov = curve_fit(hill_equation, x_data + 1e-9, y_data, p0=p0, bounds=bounds, maxfev=5000)
        top_fit, bot_fit, ec50_fit, slope_fit = popt
        
        st.success(f"모델 피팅 성공!")
        
        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_data, y_data, color='black', alpha=0.6, label='Observed Data', zorder=5)
        
        x_curve = np.logspace(np.log10(max(min(x_data[x_data>0]), 0.1)), np.log10(max(x_data)), 200)
        y_curve = hill_equation(x_curve, *popt)
        ax.plot(x_curve, y_curve, color='blue', linewidth=2, label='Fitted Curve')
        
        ax.axhline(50, color='red', linestyle='--', alpha=0.5)
        ax.axvline(ec50_fit, color='red', linestyle='--', alpha=0.5, label=f'EC50: {ec50_fit:.2f}')

        ax.set_xscale('log')
        ax.set_xlabel("Concentration (mg/L) [Log Scale]", fontsize=12)
        ax.set_ylabel("Response (%)", fontsize=12)
        ax.set_title("Dose-Response Curve (OECD TG)", fontsize=14)
        ax.set_ylim(-5, 110)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        st.pyplot(fig)

        # EC5 ~ EC95 테이블 산출
        st.write("#### 📋 독성값 상세 산출표 (EC5 ~ EC95)")
        ec_results = []
        for level in range(5, 100, 5):
            calc_conc = inverse_hill(level, top_fit, bot_fit, ec50_fit, slope_fit)
            ec_results.append({
                'Level': f"EC{level} / LC{level}",
                'Response(%)': level,
                'Calc. Conc (mg/L)': calc_conc
            })
        
        res_df = pd.DataFrame(ec_results)
        st.dataframe(
            res_df.style.highlight_between(left=49, right=51, axis=1, props='font-weight:bold; background-color:#ffffcc;')
            .format({"Calc. Conc (mg/L)": "{:.4f}"})
        )
        
    except Exception as e:
        st.error(f"곡선 피팅 실패: {e}")

# -----------------------------------------------------------------------------
# [메인 실행부]
# -----------------------------------------------------------------------------
analysis_type = st.sidebar.radio(
    "분석할 실험을 선택하세요",
    ["🟢 조류 성장저해 (Algae)", "🦐 물벼룩 유영저해 (Daphnia)", "🐟 어류 급성독성 (Fish)"]
)

st.sidebar.markdown("---")
data_source = st.sidebar.radio("데이터 소스", ["예제 데이터 사용", "CSV 업로드 (구현 예정)"])

if data_source == "예제 데이터 사용":
    df_main, y_col, y_name = get_example_data(analysis_type)
    st.write(f"### 선택된 실험: {analysis_type}")
    with st.expander("원본 데이터 보기"):
        st.dataframe(df_main)
else:
    st.info("CSV 업로드 기능은 준비 중입니다.")
    st.stop()

tab1, tab2 = st.tabs(["📊 통계 분석 (NOEC/LOEC)", "📈 독성값 산출 (ECx/LCx)"])

with tab1:
    perform_detailed_stats(df_main, y_col, y_name)

with tab2:
    calculate_dose_response(df_main, y_col)        is_homogeneous = False

    # (3) 결과 요약
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### 정규성 (Shapiro-Wilk)")
        st.dataframe(pd.DataFrame(norm_res))
    with col2:
        st.write("#### 등분산성 & 분석 방법 선정")
        st.write(f"- Levene P-value: **{l_p:.4f}** ({'등분산' if is_homogeneous else '이분산'})")
        if not is_normal:
            st.warning("👉 **비모수 검정 (Kruskal-Wallis)** 채택")
            test_type = "non-param"
        else:
            st.success("👉 **모수 검정 (ANOVA)** 채택")
            test_type = "param"

    # (4) 가설 검정 및 사후 검정
    st.write("#### 유의성 검정 결과 (Control vs Treatment)")
    comparisons = []
    noec, loec = max(concentrations), None 

    alpha = 0.05 / (len(concentrations) - 1) if len(concentrations) > 1 else 0.05

    for conc in concentrations:
        if conc == 0: continue
        
        is_sig = False
        p_val = 1.0
        method = ""

        if test_type == "non-param":
            u, p_val = stats.mannwhitneyu(control_group, groups[conc], alternative='two-sided')
            method = "Mann-Whitney"
        else:
            t, p_val = stats.ttest_ind(control_group, groups[conc], equal_var=is_homogeneous)
            method = "Welch's t-test" if not is_homogeneous else "t-test"

        is_sig = p_val < alpha
        
        comparisons.append({
            '비교 농도': conc, 
            'Method': method, 
            'P-value': f"{p_val:.4f}", 
            'Significance': '🚨 유의차 있음 (LOEC 후보)' if is_sig else '✅ 차이 없음'
        })

        if is_sig:
            if loec is None: loec = conc 
        else:
            if loec is None: noec = conc

    st.dataframe(pd.DataFrame(comparisons))
    st.info(f"📍 **결론: NOEC = {noec} mg/L, LOEC = {loec if loec else '> ' + str(max(concentrations))} mg/L**")

# -----------------------------------------------------------------------------
# [모듈 3] 용량-반응 곡선 및 ECx/LCx 전구간 산출 (Hill Equation)
# -----------------------------------------------------------------------------
def hill_equation(x, top, bottom, ec50, hill_slope):
    return bottom + (top - bottom) / (1 + (x / ec50)**(-hill_slope))

def inverse_hill(y, top, bottom, ec50, hill_slope):
    if y >= top: return np.inf
    if y <= bottom: return 0
    return ec50 * (( (top - bottom) / (y - bottom) ) - 1)**(1 / -hill_slope)

def calculate_dose_response(df, endpoint_col):
    st.markdown("### 📈 2. 농도-반응 곡선 및 ECx/LCx 산출")
    
    x_data = df['농도(mg/L)'].values
    y_data = df[endpoint_col].values

    # 초기 추정값 (Top=100, Bottom=0, EC50=Median, Slope=2)
    p0 = [100, 0, np.median(x_data[x_data > 0]), 2]
    bounds = ([90, -10, 0.0001, 0.1], [110, 10, np.inf, 20])

    try:
        popt, pcov = curve_fit(hill_equation, x_data + 1e-9, y_data, p0=p0, bounds=bounds, maxfev=5000)
        top_fit, bot_fit, ec50_fit, slope_fit = popt
        
        st.success(f"모델 피팅 성공!")
        
        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_data, y_data, color='black', alpha=0.6, label='Observed Data', zorder=5)
        
        x_curve = np.logspace(np.log10(max(min(x_data[x_data>0]), 0.1)), np.log10(max(x_data)), 200)
        y_curve = hill_equation(x_curve, *popt)
        ax.plot(x_curve, y_curve, color='blue', linewidth=2, label='Fitted Curve')
        
        ax.axhline(50, color='red', linestyle='--', alpha=0.5)
        ax.axvline(ec50_fit, color='red', linestyle='--', alpha=0.5, label=f'EC50: {ec50_fit:.2f}')

        ax.set_xscale('log')
        ax.set_xlabel("Concentration (mg/L) [Log Scale]", fontsize=12)
        ax.set_ylabel("Response (%)", fontsize=12)
        ax.set_title("Dose-Response Curve (OECD TG)", fontsize=14)
        ax.set_ylim(-5, 110)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        st.pyplot(fig)

        # EC5 ~ EC95 테이블 산출
        st.write("#### 📋 독성값 상세 산출표 (EC5 ~ EC95)")
        ec_results = []
        for level in range(5, 100, 5):
            calc_conc = inverse_hill(level, top_fit, bot_fit, ec50_fit, slope_fit)
            ec_results.append({
                'Level': f"EC{level} / LC{level}",
                'Response(%)': level,
                'Calc. Conc (mg/L)': calc_conc
            })
        
        res_df = pd.DataFrame(ec_results)
        st.dataframe(
            res_df.style.highlight_between(left=49, right=51, axis=1, props='font-weight:bold; background-color:#ffffcc;')
            .format({"Calc. Conc (mg/L)": "{:.4f}"})
        )
        
    except Exception as e:
        st.error(f"곡선 피팅 실패: {e}")

# -----------------------------------------------------------------------------
# [메인 실행부]
# -----------------------------------------------------------------------------
analysis_type = st.sidebar.radio(
    "분석할 실험을 선택하세요",
    ["🟢 조류 성장저해 (Algae)", "🦐 물벼룩 유영저해 (Daphnia)", "🐟 어류 급성독성 (Fish)"]
)

st.sidebar.markdown("---")
data_source = st.sidebar.radio("데이터 소스", ["예제 데이터 사용", "CSV 업로드 (구현 예정)"])

if data_source == "예제 데이터 사용":
    df_main, y_col, y_name = get_example_data(analysis_type)
    st.write(f"### 선택된 실험: {analysis_type}")
    with st.expander("원본 데이터 보기"):
        st.dataframe(df_main)
else:
    st.info("CSV 업로드 기능은 준비 중입니다.")
    st.stop()

tab1, tab2 = st.tabs(["📊 통계 분석 (NOEC/LOEC)", "📈 독성값 산출 (ECx/LCx)"])

with tab1:
    perform_detailed_stats(df_main, y_col, y_name)

with tab2:
    calculate_dose_response(df_main, y_col)    col1, col2 = st.columns(2)
    with col1:
        st.write("#### 정규성 (Shapiro-Wilk)")
        st.dataframe(pd.DataFrame(norm_res))
    with col2:
        st.write("#### 등분산성 & 분석 방법 선정")
        st.write(f"- Levene P-value: **{l_p:.4f}** ({'등분산' if is_homogeneous else '이분산'})")
        if not is_normal:
            st.warning("👉 **비모수 검정 (Kruskal-Wallis)** 채택")
            test_type = "non-param"
        else:
            st.success("👉 **모수 검정 (ANOVA)** 채택")
            test_type = "param"

    # (4) 가설 검정 및 사후 검정
    st.write("#### 유의성 검정 결과 (Control vs Treatment)")
    comparisons = []
    noec, loec = max(concentrations), None # 초기화

    # Alpha 보정 (Bonferroni)
    alpha = 0.05 / (len(concentrations) - 1) if len(concentrations) > 1 else 0.05

    for conc in concentrations:
        if conc == 0: continue
        
        is_sig = False
        p_val = 1.0
        method = ""

        if test_type == "non-param":
            # Mann-Whitney U
            u, p_val = stats.mannwhitneyu(control_group, groups[conc], alternative='two-sided')
            method = "Mann-Whitney"
        else:
            # T-test
            t, p_val = stats.ttest_ind(control_group, groups[conc], equal_var=is_homogeneous)
            method = "Welch's t-test" if not is_homogeneous else "t-test"

        is_sig = p_val < alpha
        
        # [수정된 부분] f-string syntax error 해결 및 결과 저장
        comparisons.append({
            '비교 농도': conc, 
            'Method': method, 
            'P-value': f"{p_val:.4f}", 
            'Significance': '🚨 유의차 있음 (LOEC 후보)' if is_sig else '✅ 차이 없음'
        })

        # NOEC/LOEC 결정 로직
        if is_sig:
            if loec is None: loec = conc # 첫 유의차가 나온 농도가 LOEC
        else:
            # 유의차가 없고, 아직 LOEC가 안나왔다면 NOEC 갱신
            if loec is None: noec = conc

    st.dataframe(pd.DataFrame(comparisons))
    
    st.info(f"📍 **결론: NOEC = {noec} mg/L, LOEC = {loec if loec else '> ' + str(max(concentrations))} mg/L**")

# -----------------------------------------------------------------------------
# [모듈 3] 용량-반응 곡선 및 ECx/LCx 전구간 산출 (Hill Equation)
# -----------------------------------------------------------------------------
def hill_equation(x, top, bottom, ec50, hill_slope):
    """
    4-Parameter Logistic Equation (Hill Equation)
    x: 농도
    top: 최대 반응 (보통 100)
    bottom: 최소 반응 (보통 0)
    ec50: 50% 반응 농도
    hill_slope: 기울기
    """
    # x가 0일 때 log 계산 오류 방지를 위해 매우 작은 값 더함 (시각화용 아님, 계산용)
    return bottom + (top - bottom) / (1 + (x / ec50)**(-hill_slope))

def inverse_hill(y, top, bottom, ec50, hill_slope):
    """Hill 식의 역함수: 반응값(y)을 넣으면 농도(x)를 반환"""
    # y가 범위 밖이면 계산 불가
    if y >= top: return np.inf
    if y <= bottom: return 0
    return ec50 * (( (top - bottom) / (y - bottom) ) - 1)**(1 / -hill_slope)

def calculate_dose_response(df, endpoint_col):
    st.markdown("### 📈 2. 농도-반응 곡선 및 ECx/LCx 산출")
    
    x_data = df['농도(mg/L)'].values
    y_data = df[endpoint_col].values

    # 초기 추정값 (p0): [top, bottom, ec50, slope]
    # Top은 100 근처, Bottom은 0 근처, EC50은 중간 농도, Slope는 양수/음수 가정
    # 여기서는 "농도가 높을수록 반응(%)이 커진다"고 가정 (예: 치사율, 저해율)
    # 따라서 Slope는 양수여야 함.
    p0 = [100, 0, np.median(x_data[x_data > 0]), 2]
    
    # 경계 조건 (Bounds): Top(90~110), Bottom(-10~10), EC50(>0), Slope(>0)
    bounds = ([90, -10, 0.0001, 0.1], [110, 10, np.inf, 20])

    try:
        popt, pcov = curve_fit(hill_equation, x_data + 1e-9, y_data, p0=p0, bounds=bounds, maxfev=5000)
        top_fit, bot_fit, ec50_fit, slope_fit = popt
        
        st.success(f"모델 피팅 성공! (R-squared 계산 생략)")
        
        # 1. 그래프 그리기 (OECD Style)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 원본 데이터 점
        ax.scatter(x_data, y_data, color='black', alpha=0.6, label='Observed Data', zorder=5)
        
        # 피팅 곡선
        x_curve = np.logspace(np.log10(max(min(x_data[x_data>0]), 0.1)), np.log10(max(x_data)), 200)
        y_curve = hill_equation(x_curve, *popt)
        ax.plot(x_curve, y_curve, color='blue', linewidth=2, label='Fitted Curve (Hill Model)')
        
        # 50% 지점 표시
        ax.axhline(50, color='red', linestyle='--', alpha=0.5)
        ax.axvline(ec50_fit, color='red', linestyle='--', alpha=0.5, label=f'EC50: {ec50_fit:.2f}')

        ax.set_xscale('log') # OECD는 보통 로그 스케일 권장
        ax.set_xlabel("Concentration (mg/L) [Log Scale]", fontsize=12)
        ax.set_ylabel("Response (%)", fontsize=12)
        ax.set_title("Dose-Response Curve (OECD TG)", fontsize=14)
        ax.set_ylim(-5, 110)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
        st.pyplot(fig)

        # 2. EC5 ~ EC95 테이블 산출 (요청사항)
        st.write("#### 📋 독성값 상세 산출표 (EC5 ~ EC95)")
        
        ec_results = []
        # 5% 부터 95% 까지 5단위로 반복
        target_levels = range(5, 100, 5) 
        
        for level in target_levels:
            calc_conc = inverse_hill(level, top_fit, bot_fit, ec50_fit, slope_fit)
            ec_results.append({
                'Level': f"EC{level} / LC{level}",
                'Response(%)': level,
                'Calc. Conc (mg/L)': calc_conc
            })
        
        res_df = pd.DataFrame(ec_results)
        
        # 주요 값 하이라이트 표시
        st.dataframe(
            res_df.style.highlight_between(left=49, right=51, axis=1, props='font-weight:bold; background-color:#ffffcc;')
            .format({"Calc. Conc (mg/L)": "{:.4f}"})
        )
        
    except Exception as e:
        st.error(f"곡선 피팅에 실패했습니다. 데이터가 불규칙하거나 부족합니다.\nError: {e}")
        st.write("선형 보간법(Linear Interpolation) 결과를 대신 확인하세요.")

# -----------------------------------------------------------------------------
# [메인 실행부]
# -----------------------------------------------------------------------------
# 사이드바 설정
analysis_type = st.sidebar.radio(
    "분석할 실험을 선택하세요",
    ["🟢 조류 성장저해 (Algae)", "🦐 물벼룩 유영저해 (Daphnia)", "🐟 어류 급성독성 (Fish)"]
)

st.sidebar.markdown("---")
data_source = st.sidebar.radio("데이터 소스", ["예제 데이터 사용", "CSV 업로드 (구현 예정)"])

# 데이터 로드
if data_source == "예제 데이터 사용":
    df_main, y_col, y_name = get_example_data(analysis_type)
    st.write(f"### 선택된 실험: {analysis_type}")
    with st.expander("원본 데이터 보기"):
        st.dataframe(df_main)
else:
    st.info("CSV 업로드 기능은 준비 중입니다.")
    st.stop()

# 탭 구성
tab1, tab2 = st.tabs(["📊 통계 분석 (NOEC/LOEC)", "📈 독성값 산출 (ECx/LCx)"])

with tab1:
    perform_detailed_stats(df_main, y_col, y_name)

with tab2:
    calculate_dose_response(df_main, y_col)        df['수율'] = df['최종 세포수 (cells/mL)'] - init_cells
        df['비성장률'] = (np.log(df['최종 세포수 (cells/mL)']) - np.log(init_cells)) / (duration/24)
        
        # ---------------------------------------------------------
        # [복구됨] 생물량 및 성장률 분포 그래프 (Boxplot)
        # ---------------------------------------------------------
        st.divider()
        st.subheader("📊 데이터 분포 시각화 (Boxplot)")
        
        fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        concs = sorted(df['농도(mg/L)'].unique())
        yield_data = [df[df['농도(mg/L)'] == c]['수율'] for c in concs]
        rate_data = [df[df['농도(mg/L)'] == c]['비성장률'] for c in concs]
        
        # 수율 그래프
        ax1.boxplot(yield_data, labels=concs, patch_artist=True, boxprops=dict(facecolor='#D1E8E2'))
        ax1.set_title('Yield (Biomass)')
        ax1.set_xlabel('Concentration (mg/L)')
        ax1.set_ylabel('Yield (Cell Increase)')
        ax1.grid(axis='y', linestyle=':', alpha=0.7)

        # 비성장률 그래프
        ax2.boxplot(rate_data, labels=concs, patch_artist=True, boxprops=dict(facecolor='#F2D7D5'))
        ax2.set_title('Specific Growth Rate')
        ax2.set_xlabel('Concentration (mg/L)')
        ax2.set_ylabel('Growth Rate (1/day)')
        ax2.grid(axis='y', linestyle=':', alpha=0.7)

        st.pyplot(fig_dist)
        st.divider()
        
        # 탭 구성 (상세 통계 및 EC50)
        tab1, tab2 = st.tabs(["📈 비성장률(Rate) 분석", "📉 수율(Yield) 분석"])
        
        def show_results(target_col, name, ec_label):
            # 1. 상세 통계 (NOEC/LOEC)
            perform_detailed_stats(df, target_col, name)
            
            # 2. EC50 산출
            control_mean = df[df['농도(mg/L)']==0][target_col].mean()
            ec50, r2, method, plot_info = calculate_point_estimate(df, target_col, control_mean, ec_label)
            
            st.markdown(f"#### 5. {ec_label} 산출 결과")
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric(f"{ec_label}", f"{ec50:.4f} mg/L" if ec50 else "산출 불가")
            cm2.metric("적용 모델", method)
            cm3.metric("R²", f"{r2:.4f}" if r2 > 0 else "-")
            
            fig, ax = plt.subplots(figsize=(6, 4))
            if plot_info['type'] == 'probit':
                x = plot_info['x']
                slope = plot_info['slope']
                intercept = plot_info['intercept']
                x_line = np.linspace(min(x), max(x), 100)
                
                ax.scatter(x, plot_info['y'], label='Data')
                ax.plot(x_line, slope*x_line + intercept, color='red', label='Probit Fit')
                ax.set_xlabel('Log Concentration')
                ax.set_ylabel('Probit (Inhibition)')
                if plot_info['ec50']:
                    ax.axvline(plot_info['ec50'], color='green', linestyle='--', label='50% Effect')
            else:
                d = plot_info['data']
                ax.plot(d['농도(mg/L)'], d['Inhibition'], marker='o', label='Linear Interp')
                ax.axhline(0.5, color='red', linestyle='--', label='50% Cutoff')
                if plot_info['ec50']:
                    ax.axvline(plot_info['ec50'], color='green', linestyle='--')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with tab1:
            show_results('비성장률', '비성장률', 'ErC50')
        with tab2:
            show_results('수율', '수율', 'EyC50')

# -----------------------------------------------------------------------------
# [분석 실행 함수] 어류/물벼룩
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
    
    if st.button("분석 실행"):
        df = df_input.copy()
        
        st.subheader(f"📊 {label} 산출 결과")
        ec50, r2, method, plot_info = calculate_point_estimate(df, '반응 수', 0, label)
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{label}", f"{ec50:.4f} mg/L" if ec50 else "산출 불가")
        c2.metric("계산 방식", method)
        c3.metric("R²", f"{r2:.4f}" if r2 > 0 else "-")
        
        fig, ax = plt.subplots()
        if plot_info['type'] == 'probit':
            ax.scatter(plot_info['x'], plot_info['y'], label='Data')
            x_line = np.linspace(min(plot_info['x']), max(plot_info['x']), 100)
            ax.plot(x_line, plot_info['slope']*x_line + plot_info['intercept'], color='red')
            ax.set_xlabel('Log Concentration')
            ax.set_ylabel('Probit')
        else:
            d = plot_info['data']
            ax.plot(d['농도(mg/L)'], d['Inhibition'], marker='o')
            ax.set_xlabel('Concentration')
            ax.set_ylabel('Response Rate')
        
        ax.legend()
        st.pyplot(fig)


# -----------------------------------------------------------------------------
# 메인 실행
# -----------------------------------------------------------------------------
if "조류" in analysis_type:
    run_algae_analysis()
elif "물벼룩" in analysis_type:
    run_animal_analysis("🦐 물벼룩 급성 유영저해", "EC50")
elif "어류" in analysis_type:
    run_animal_analysis("🐟 어류 급성 독성", "LC50")
