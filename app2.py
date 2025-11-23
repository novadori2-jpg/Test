import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# [공통] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생태독성 통합 분석기 (Pro)", page_icon="🧬", layout="wide")

st.title("🧬 생태독성 통합 분석 어플리케이션 (Pro)")
st.markdown("""
이 앱은 다음 세 가지 실험에 대한 전문 통계 분석을 지원합니다:
1. **조류 (Algae):** 생물량(수율) 및 성장률 분포 시각화, NOEC/LOEC, ErC50/EyC50 산출
2. **물벼룩 (Daphnia):** 급성 유영저해 시험 (EC50)
3. **어류 (Fish):** 급성 독성 시험 (LC50)
""")
st.divider()

# 사이드바에서 실험 종류 선택
analysis_type = st.sidebar.radio(
    "분석할 실험을 선택하세요",
    ["🟢 조류 성장저해 (Algae)", "🦐 물벼룩 유영저해 (Daphnia)", "🐟 어류 급성독성 (Fish)"]
)

# -----------------------------------------------------------------------------
# [핵심 함수] 통계 및 EC50 산출 통합 함수 (조류용)
# -----------------------------------------------------------------------------
def analyze_algae_endpoint(df, endpoint_col, endpoint_name, ec_label):
    st.markdown(f"### 📊 {endpoint_name} 상세 분석 ({ec_label})")
    
    groups = df.groupby('농도(mg/L)')[endpoint_col].apply(list)
    concentrations = sorted(groups.keys())
    control_group = groups[0]
    control_mean = np.mean(control_group)

    # --- A. 통계적 유의성 검정 (NOEC/LOEC) ---
    st.markdown("#### 1. 유의성 검정 (NOEC / LOEC)")
    
    # 등분산성 (Levene)
    data_list = [groups[c] for c in concentrations]
    if len(data_list) < 2:
        st.error("데이터 그룹이 충분하지 않습니다.")
        return

    l_stat, l_p = stats.levene(*data_list)
    
    # ANOVA
    f_stat, f_p = stats.f_oneway(*data_list)

    noec = 0
    loec = None
    
    # 결과 테이블용 리스트
    stat_summary = []
    
    if f_p < 0.05:
        st.write(f"One-way ANOVA 결과 유의차 있음 (P={f_p:.4f}). 다중비교를 수행합니다.")
        alpha = 0.05 / (len(concentrations) - 1) # Bonferroni

        for conc in concentrations:
            if conc == 0: continue
            t_stat, t_p = stats.ttest_ind(control_group, groups[conc], equal_var=(l_p > 0.05))
            is_sig = t_p < alpha
            
            stat_summary.append({
                '비교 농도': conc,
                'T-value': f"{t_stat:.2f}",
                'P-value': f"{t_p:.4f}",
                '판정': '🚨 유의차 있음' if is_sig else '✅ 차이 없음'
            })

            if is_sig and loec is None:
                loec = conc
            if not is_sig:
                noec = conc
        
        st.table(pd.DataFrame(stat_summary))
    else:
        st.info(f"ANOVA 결과 유의한 차이가 없습니다. (P={f_p:.4f})")
        noec = max(concentrations)
    
    c1, c2 = st.columns(2)
    c1.metric(f"{endpoint_name} NOEC", f"{noec} mg/L")
    c2.metric(f"{endpoint_name} LOEC", f"{loec if loec else '> Max'} mg/L")

    st.divider()

    # --- B. 독성값 산출 (EC50) ---
    st.markdown(f"#### 2. {ec_label} 산출 (저해율 기반 Probit)")

    try:
        dose_resp = df.groupby('농도(mg/L)')[endpoint_col].mean().reset_index()
        dose_resp = dose_resp[dose_resp['농도(mg/L)'] > 0].copy()

        # 저해율 계산 (%)
        dose_resp['Inhibition'] = (control_mean - dose_resp[endpoint_col]) / control_mean
        dose_resp['Inhibition_adj'] = dose_resp['Inhibition'].clip(0.001, 0.999)
        dose_resp['Probit'] = stats.norm.ppf(dose_resp['Inhibition_adj'])
        dose_resp['Log_Conc'] = np.log10(dose_resp['농도(mg/L)'])

        slope, intercept, r_val, p_val, std_err = stats.linregress(dose_resp['Log_Conc'], dose_resp['Probit'])

        log_ec50 = -intercept / slope
        ec50_val = 10 ** log_ec50

        c1, c2 = st.columns(2)
        c1.metric(f"추정 {ec_label}", f"{ec50_val:.4f} mg/L")
        c2.metric("결정계수 ($R^2$)", f"{r_val**2:.4f}")

        # 회귀 그래프
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(dose_resp['Log_Conc'], dose_resp['Probit'], label='Data Points', color='blue')
        x_range = np.linspace(dose_resp['Log_Conc'].min(), dose_resp['Log_Conc'].max(), 100)
        ax.plot(x_range, slope*x_range + intercept, color='red', label='Regression Line')
        ax.axhline(0, color='green', linestyle='--', label='50% Inhibition')
        ax.axvline(log_ec50, color='green', linestyle='--')
        ax.set_xlabel('Log Concentration')
        ax.set_ylabel('Probit (Inhibition)')
        ax.set_title(f"{ec_label} Regression Analysis")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"{ec_label} 산출 불가 (데이터 경향성 확인 필요): {e}")


# -----------------------------------------------------------------------------
# [기능 1] 조류 성장저해 분석 (메인)
# -----------------------------------------------------------------------------
def run_algae_analysis():
    st.header("🟢 조류 성장저해 시험 (OECD TG 201)")
    st.info("초기 세포수와 최종 세포수를 입력하면 **생물량(수율)**과 **비성장률**을 계산하고 분포를 시각화합니다.")

    with st.expander("⚙️ 실험 조건 설정 (클릭하여 열기)", expanded=True):
        col_s1, col_s2 = st.columns(2)
        init_cells = col_s1.number_input("초기 세포수 (cells/mL)", value=10000, step=1000, format="%d")
        duration_hour = col_s2.number_input("배양 시간 (시간)", value=72, step=24)

    if 'algae_data_v2' not in st.session_state:
        st.session_state.algae_data_v2 = pd.DataFrame({
            '농도(mg/L)': [0, 0, 0, 10, 10, 10, 32, 32, 32, 100, 100, 100],
            '최종 세포수 (cells/mL)': [
                1000000, 1050000, 980000,
                900000, 880000, 910000,
                500000, 480000, 520000,
                150000, 140000, 160000
            ]
        })

    st.subheader("📝 데이터 입력")
    df_input = st.data_editor(
        st.session_state.algae_data_v2, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={"최종 세포수 (cells/mL)": st.column_config.NumberColumn(format="%d")}
    )

    if st.button("분석 실행 (그래프 및 통계)"):
        if df_input.empty:
            st.error("데이터가 없습니다.")
            return

        # 데이터 계산
        df = df_input.copy()
        df['수율'] = df['최종 세포수 (cells/mL)'] - init_cells
        # 비성장률 (일 단위)
        df['비성장률'] = (np.log(df['최종 세포수 (cells/mL)']) - np.log(init_cells)) / (duration_hour / 24)

        # ---------------------------------------------------------
        # [추가됨] 생물량 및 성장률 분포 그래프 (Boxplot)
        # ---------------------------------------------------------
        st.divider()
        st.subheader("📊 생물량 및 성장률 분포 (Boxplot)")
        st.markdown("각 농도별 데이터의 분포(평균 및 편차)를 시각화합니다.")
        
        # Boxplot 그리기
        fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 그래프를 그리기 위해 데이터를 리스트 형태로 변환
        concs = sorted(df['농도(mg/L)'].unique())
        yield_data = [df[df['농도(mg/L)'] == c]['수율'] for c in concs]
        rate_data = [df[df['농도(mg/L)'] == c]['비성장률'] for c in concs]
        
        # 1. 수율(Biomass) 그래프
        ax1.boxplot(yield_data, labels=concs, patch_artist=True, boxprops=dict(facecolor='#D1E8E2'))
        ax1.set_title('Yield (Biomass) Distribution')
        ax1.set_xlabel('Concentration (mg/L)')
        ax1.set_ylabel('Yield (Cell increase)')
        ax1.grid(axis='y', linestyle=':', alpha=0.7)

        # 2. 성장률 그래프
        ax2.boxplot(rate_data, labels=concs, patch_artist=True, boxprops=dict(facecolor='#F2D7D5'))
        ax2.set_title('Specific Growth Rate Distribution')
        ax2.set_xlabel('Concentration (mg/L)')
        ax2.set_ylabel('Growth Rate (1/day)')
        ax2.grid(axis='y', linestyle=':', alpha=0.7)

        st.pyplot(fig_dist)
        st.caption("박스(Box)는 데이터의 50% 범위를, 가운데 선은 중앙값(Median)을 나타냅니다.")
        st.divider()

        # ---------------------------------------------------------
        # 결과 탭 (통계 및 EC50)
        # ---------------------------------------------------------
        tab1, tab2 = st.tabs(["📈 비성장률 분석 (ErC50)", "📉 수율 분석 (EyC50)"])
        
        with tab1:
            analyze_algae_endpoint(df, '비성장률', '비성장률(Growth Rate)', 'ErC50')
            
        with tab2:
            analyze_algae_endpoint(df, '수율', '수율(Yield, 생물량)', 'EyC50')


# -----------------------------------------------------------------------------
# [기능 2] 어류/물벼룩 Probit 분석 (기존 유지)
# -----------------------------------------------------------------------------
def run_probit_analysis(test_name, value_label):
    st.header(f"{test_name} 분석")
    st.info(f"농도별 반응 수(사망/유영저해)를 입력하여 {value_label}를 산출합니다.")

    key_name = f"data_{value_label}"
    if key_name not in st.session_state:
        st.session_state[key_name] = pd.DataFrame({
            '농도(mg/L)': [0, 6.25, 12.5, 25.0, 50.0, 100.0],
            '총 개체수': [10, 10, 10, 10, 10, 10],
            '반응 수': [0, 0, 1, 5, 9, 10]
        })

    edited_df = st.data_editor(st.session_state[key_name], num_rows="dynamic", use_container_width=True)

    if st.button(f"{value_label} 계산하기"):
        try:
            df = edited_df.copy()
            df_calc = df[df['농도(mg/L)'] > 0].copy()

            if len(df_calc) < 3:
                st.warning("최소 3개 이상의 농도 데이터가 필요합니다.")
                return

            df_calc['반응률'] = df_calc['반응 수'] / df_calc['총 개체수']
            df_calc['반응률_보정'] = df_calc['반응률'].clip(0.001, 0.999)
            df_calc['Probit'] = stats.norm.ppf(df_calc['반응률_보정'])
            df_calc['Log_농도'] = np.log10(df_calc['농도(mg/L)'])

            slope, intercept, r_value, p_value, std_err = stats.linregress(df_calc['Log_농도'], df_calc['Probit'])

            log_50 = -intercept / slope
            result_val = 10 ** log_50

            c1, c2 = st.columns(2)
            c1.metric(f"{value_label} 결과", f"{result_val:.4f} mg/L")
            c2.metric("결정계수 (R²)", f"{r_value**2:.4f}")

            fig, ax = plt.subplots()
            ax.scatter(df_calc['Log_농도'], df_calc['Probit'], label='Data')
            x_range = np.linspace(df_calc['Log_농도'].min(), df_calc['Log_농도'].max(), 100)
            ax.plot(x_range, slope * x_range + intercept, color='red', label='Regression')
            ax.axhline(0, color='green', linestyle='--', label='50% Response')
            ax.axvline(log_50, color='green', linestyle='--')
            ax.set_xlabel('Log Concentration')
            ax.set_ylabel('Probit')
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"계산 오류: {e}")

# -----------------------------------------------------------------------------
# [메인] 실행 로직
# -----------------------------------------------------------------------------
if "조류" in analysis_type:
    run_algae_analysis()
elif "물벼룩" in analysis_type:
    run_probit_analysis("🦐 물벼룩 급성 유영저해", "EC50")
elif "어류" in analysis_type:
    run_probit_analysis("🐟 어류 급성 독성", "LC50")
