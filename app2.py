import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# [공통] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생태독성 전문 분석기 (CETIS Logic)", page_icon="🧬", layout="wide")

st.title("🧬 생태독성 전문 분석기 (Algorithm Enhanced)")
st.markdown("""
이 앱은 **CETIS/ToxCalc 등 전문 소프트웨어의 의사결정 로직**을 따릅니다.
1. **통계 검정:** 정규성/등분산성 여부에 따라 **모수(ANOVA)** 또는 **비모수(Kruskal-Wallis/Mann-Whitney)** 검정을 자동 선택합니다.
2. **독성값 산출:** **Probit** 분석을 우선 수행하며, 적합도가 낮거나 실패 시 **선형보간법(Linear Interpolation)**을 자동 적용합니다.
""")
st.divider()

analysis_type = st.sidebar.radio(
    "분석할 실험을 선택하세요",
    ["🟢 조류 성장저해 (Algae)", "🦐 물벼룩 유영저해 (Daphnia)", "🐟 어류 급성독성 (Fish)"]
)

# -----------------------------------------------------------------------------
# [핵심 로직 1] 통계적 가설 검정 (NOEC/LOEC Decision Tree)
# -----------------------------------------------------------------------------
def perform_hypothesis_test(df, endpoint_col):
    """
    데이터의 정규성/등분산성을 판단하여 적절한 통계 기법(모수/비모수)을 자동 선택
    """
    groups = df.groupby('농도(mg/L)')[endpoint_col].apply(list)
    concentrations = sorted(groups.keys())
    control_group = groups[0]
    
    # 1. 정규성 검정 (Shapiro-Wilk)
    is_normal = True
    normality_p_values = []
    for conc in concentrations:
        data = groups[conc]
        if len(data) >= 3:
            stat, p = stats.shapiro(data)
            normality_p_values.append(p)
            if p < 0.01: # 엄격한 기준 0.01 (또는 0.05)
                is_normal = False
    
    # 2. 등분산성 검정 (Levene)
    data_list = [groups[c] for c in concentrations]
    l_stat, l_p = stats.levene(*data_list)
    is_homogeneous = l_p > 0.05

    # 결과 저장용
    decision_log = []
    decision_log.append(f"- 정규성 만족 여부: {'✅ Yes' if is_normal else '❌ No'}")
    decision_log.append(f"- 등분산성 만족 여부: {'✅ Yes' if is_homogeneous else '❌ No'}")

    comparisons = []
    noec = 0
    loec = None
    
    # --- 로직 분기 (Decision Tree) ---
    
    # CASE A: 비모수 검정 (정규성 위배 시) -> Kruskal-Wallis + Mann-Whitney U
    if not is_normal:
        test_method = "비모수 검정 (Non-Parametric)"
        sub_test = "Mann-Whitney U (Bonferroni adj)"
        decision_log.append(f"👉 **{test_method}**을 수행합니다. (정규성 위배)")
        
        # Kruskal-Wallis
        k_stat, k_p = stats.kruskal(*data_list)
        decision_log.append(f"- Kruskal-Wallis P-value: {k_p:.4f}")

        if k_p < 0.05:
            alpha = 0.05 / (len(concentrations) - 1)
            for conc in concentrations:
                if conc == 0: continue
                # Mann-Whitney U test (Two-sided)
                u_stat, u_p = stats.mannwhitneyu(control_group, groups[conc], alternative='two-sided')
                is_sig = u_p < alpha
                comparisons.append({
                    '농도': conc, 'Method': 'Mann-Whitney', 'P-value': f"{u_p:.4f}", 
                    'Alpha': f"{alpha:.4f}", '판정': '🚨 유의차 있음' if is_sig else '✅ 차이 없음'
                })
                if is_sig and loec is None: loec = conc
                if not is_sig: noec = conc
        else:
             decision_log.append("- 그룹 간 유의한 차이 없음")
             noec = max(concentrations)

    # CASE B: 모수 검정 (정규성 만족) -> ANOVA + t-test
    else:
        test_method = "모수 검정 (Parametric)"
        
        # ANOVA
        f_stat, f_p = stats.f_oneway(*data_list)
        
        if f_p < 0.05:
            alpha = 0.05 / (len(concentrations) - 1)
            
            if is_homogeneous:
                sub_test = "Bonferroni t-test (Equal Var)"
                decision_log.append(f"👉 **{test_method}**을 수행합니다. (정규성/등분산성 만족)")
                decision_log.append(f"- ANOVA P-value: {f_p:.4f}")
                
                for conc in concentrations:
                    if conc == 0: continue
                    t_stat, t_p = stats.ttest_ind(control_group, groups[conc], equal_var=True)
                    is_sig = t_p < alpha
                    comparisons.append({
                        '농도': conc, 'Method': 't-test(Eq)', 'P-value': f"{t_p:.4f}", 
                        'Alpha': f"{alpha:.4f}", '판정': '🚨 유의차 있음' if is_sig else '✅ 차이 없음'
                    })
                    if is_sig and loec is None: loec = conc
                    if not is_sig: noec = conc
            else:
                sub_test = "Welch's t-test (Unequal Var)"
                decision_log.append(f"👉 **{test_method}**을 수행합니다. (등분산성 위배 -> Welch 보정)")
                decision_log.append(f"- ANOVA P-value: {f_p:.4f}")
                
                for conc in concentrations:
                    if conc == 0: continue
                    t_stat, t_p = stats.ttest_ind(control_group, groups[conc], equal_var=False)
                    is_sig = t_p < alpha
                    comparisons.append({
                        '농도': conc, 'Method': 'Welch-t', 'P-value': f"{t_p:.4f}", 
                        'Alpha': f"{alpha:.4f}", '판정': '🚨 유의차 있음' if is_sig else '✅ 차이 없음'
                    })
                    if is_sig and loec is None: loec = conc
                    if not is_sig: noec = conc
        else:
            decision_log.append(f"👉 **{test_method}** 결과 유의한 차이가 없습니다. (ANOVA P={f_p:.4f})")
            noec = max(concentrations)
            sub_test = "ANOVA"

    return decision_log, comparisons, noec, loec, test_method, sub_test


# -----------------------------------------------------------------------------
# [핵심 로직 2] EC50/LC50 산출 (Probit -> Interpolation Fallback)
# -----------------------------------------------------------------------------
def calculate_point_estimate(df, endpoint_col, control_mean, label):
    """
    1순위: Probit 분석
    2순위: 실패 시 선형보간법(Linear Interpolation)
    """
    # 데이터 전처리 (평균 반응률 계산)
    dose_resp = df.groupby('농도(mg/L)')[endpoint_col].mean().reset_index()
    dose_resp = dose_resp[dose_resp['농도(mg/L)'] > 0].copy() # 대조군 제외

    # 저해율(Inhibition) 계산: (Control - Treat) / Control
    # 만약 endpoint가 '반응 수'라면 그대로 비율 사용
    if '반응 수' in df.columns:
        # 어류/물벼룩의 경우
        total = df.groupby('농도(mg/L)')['총 개체수'].mean()[dose_resp['농도(mg/L)']].values
        dose_resp['Inhibition'] = dose_resp[endpoint_col] / total
    else:
        # 조류의 경우 (세포수, 성장률 등)
        dose_resp['Inhibition'] = (control_mean - dose_resp[endpoint_col]) / control_mean

    # --- Method 1: Probit Analysis ---
    method_used = "Probit Analysis"
    ec50_val = None
    r_squared = 0
    
    try:
        # Probit 변환
        # 0과 1은 무한대가 되므로 clip 처리
        dose_resp['Inhibition_adj'] = dose_resp['Inhibition'].clip(0.001, 0.999)
        dose_resp['Probit'] = stats.norm.ppf(dose_resp['Inhibition_adj'])
        dose_resp['Log_Conc'] = np.log10(dose_resp['농도(mg/L)'])

        # 회귀분석
        slope, intercept, r_val, p_val, std_err = stats.linregress(dose_resp['Log_Conc'], dose_resp['Probit'])
        r_squared = r_val ** 2
        
        # 적합도가 너무 낮거나(0.6 미만), 기울기가 음수(독성이상)인 경우 Fallback
        if r_squared < 0.6 or slope <= 0:
            raise ValueError("Low R-squared or Invalid slope")

        log_ec50 = -intercept / slope
        ec50_val = 10 ** log_ec50
        
        # 시각화용 데이터
        x_plot = dose_resp['Log_Conc']
        y_plot = dose_resp['Probit']
        x_line = np.linspace(min(x_plot), max(x_plot), 100)
        y_line = slope * x_line + intercept
        
        plot_info = {'type': 'probit', 'x': x_plot, 'y': y_plot, 'x_line': x_line, 'y_line': y_line, 'ec50': log_ec50}

    except Exception:
        # --- Method 2: Linear Interpolation (Fallback) ---
        method_used = "Linear Interpolation (ICp)"
        # 50% 저해율을 지나는 두 점 찾기
        # 데이터 정렬
        dose_resp = dose_resp.sort_values('Inhibition')
        
        # 0.5(50%) 바로 아래와 바로 위 찾기
        lower = dose_resp[dose_resp['Inhibition'] <= 0.5].max()
        upper = dose_resp[dose_resp['Inhibition'] >= 0.5].min()
        
        if pd.isna(lower['Inhibition']) or pd.isna(upper['Inhibition']):
            ec50_val = None # 범위 밖
        else:
            # 선형 보간 공식: x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            x1, y1 = lower['농도(mg/L)'], lower['Inhibition']
            x2, y2 = upper['농도(mg/L)'], upper['Inhibition']
            
            if y1 == y2: # 정확히 같은 경우
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
        init_cells = c1.number_input("초기 세포수", value=10000, format="%d")
        duration = c2.number_input("배양 시간 (h)", value=72)

    if 'algae_data_pro' not in st.session_state:
        st.session_state.algae_data_pro = pd.DataFrame({
            '농도(mg/L)': [0, 0, 0, 10, 10, 10, 32, 32, 32, 100, 100, 100],
            '최종 세포수 (cells/mL)': [1000000, 1050000, 980000, 900000, 880000, 910000, 500000, 480000, 520000, 150000, 140000, 160000]
        })
    
    df_input = st.data_editor(st.session_state.algae_data_pro, num_rows="dynamic", use_container_width=True)
    
    if st.button("자동 분석 실행"):
        df = df_input.copy()
        df['수율'] = df['최종 세포수 (cells/mL)'] - init_cells
        df['비성장률'] = (np.log(df['최종 세포수 (cells/mL)']) - np.log(init_cells)) / (duration/24)
        
        # 탭 구성
        tab1, tab2 = st.tabs(["📈 비성장률(Rate)", "📉 수율(Yield)"])
        
        # 내부 분석 함수
        def analyze_endpoint(target_col, target_name, ec_label):
            st.markdown(f"### {target_name} 분석")
            
            # 1. NOEC/LOEC (Decision Tree)
            logs, comps, noec, loec, method, sub_method = perform_hypothesis_test(df, target_col)
            
            with st.expander(f"📋 통계적 검정 과정 확인 ({method})", expanded=True):
                for log in logs: st.write(log)
                if comps: st.table(pd.DataFrame(comps))
            
            c1, c2 = st.columns(2)
            c1.metric(f"{target_name} NOEC", f"{noec} mg/L", help=f"사용된 검정법: {sub_method}")
            c2.metric(f"{target_name} LOEC", f"{loec if loec else '>Max'} mg/L")
            
            # 2. EC50 (Fallback Logic)
            control_mean = df[df['농도(mg/L)']==0][target_col].mean()
            ec50, r2, calc_method, plot_info = calculate_point_estimate(df, target_col, control_mean, ec_label)
            
            st.divider()
            st.markdown(f"#### {ec_label} 산출 결과")
            cm1, cm2, cm3 = st.columns(3)
            cm1.metric(f"{ec_label}", f"{ec50:.4f} mg/L" if ec50 else "산출 불가")
            cm2.metric("계산 방식", calc_method)
            cm3.metric("결정계수 (R²)", f"{r2:.4f}" if r2 > 0 else "-")
            
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(6, 4))
            if plot_info['type'] == 'probit':
                ax.scatter(plot_info['x'], plot_info['y'], label='Data')
                ax.plot(plot_info['x_line'], plot_info['y_line'], color='red', label='Probit Fit')
                ax.set_xlabel('Log Concentration')
                ax.set_ylabel('Probit (Inhibition)')
                if plot_info['ec50']:
                    ax.axvline(plot_info['ec50'], color='green', linestyle='--', label='50% Effect')
            else:
                # Interpolation Graph
                d = plot_info['data']
                ax.plot(d['농도(mg/L)'], d['Inhibition'], marker='o', linestyle='-', label='Linear Interp')
                ax.set_xlabel('Concentration')
                ax.set_ylabel('Inhibition (0~1)')
                if plot_info['ec50']:
                    ax.axvline(plot_info['ec50'], color='green', linestyle='--', label='EC50')
                    ax.axhline(0.5, color='gray', linestyle=':')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with tab1: analyze_endpoint('비성장률', '비성장률', 'ErC50')
        with tab2: analyze_endpoint('수율', '수율', 'EyC50')

# -----------------------------------------------------------------------------
# [분석 실행 함수] 어류/물벼룩
# -----------------------------------------------------------------------------
def run_animal_analysis(test_name, label):
    st.header(f"{test_name}")
    
    key = f"data_{label}"
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame({
            '농도(mg/L)': [0, 6.25, 12.5, 25.0, 50.0, 100.0],
            '총 개체수': [10, 10, 10, 10, 10, 10],
            '반응 수': [0, 0, 1, 5, 9, 10]
        })
    
    df_input = st.data_editor(st.session_state[key], num_rows="dynamic", use_container_width=True)
    
    if st.button("분석 실행"):
        df = df_input.copy()
        
        # EC50 / LC50 산출 (Fallback Logic 적용)
        # 어류/물벼룩은 반응 수 자체가 endpoint이므로 처리 방식이 약간 다름
        # calculate_point_estimate 함수는 비율(Inhibition)을 기대하므로 컬럼명을 맞춰줌
        
        st.divider()
        st.subheader(f"📊 {label} 산출 결과")
        
        # Probit or Interpolation
        # control_mean은 의미 없지만 함수 인자 맞추기 위해 0 전달 (함수 내부에서 반응 수 처리 로직 분기)
        ec50, r2, method, plot_info = calculate_point_estimate(df, '반응 수', 0, label)
        
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{label}", f"{ec50:.4f} mg/L" if ec50 else "산출 불가")
        c2.metric("계산 방식", method)
        c3.metric("R² (Probit일 경우)", f"{r2:.4f}" if r2 > 0 else "-")
        
        # 그래프
        fig, ax = plt.subplots()
        if plot_info['type'] == 'probit':
            ax.scatter(plot_info['x'], plot_info['y'], label='Data')
            ax.plot(plot_info['x_line'], plot_info['y_line'], color='red', label='Probit Model')
            ax.set_xlabel('Log Concentration')
            ax.set_ylabel('Probit')
        else:
            d = plot_info['data']
            ax.plot(d['농도(mg/L)'], d['Inhibition'], marker='o', label='Measured')
            ax.set_xlabel('Concentration')
            ax.set_ylabel('Response Rate')
            ax.axhline(0.5, color='red', linestyle='--', label='50% Effect')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
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
