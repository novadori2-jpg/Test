import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# [공통] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생태독성 통합 분석기", page_icon="🧬", layout="wide")

st.title("🧬 생태독성 통합 분석 어플리케이션")
st.markdown("""
이 앱은 다음 세 가지 실험에 대한 통계 분석을 지원합니다:
1. **조류 (Algae):** 성장저해 시험 (세포수 기반 ANOVA, NOEC/LOEC)
2. **물벼룩 (Daphnia):** 급성 유영저해 시험 (Probit 분석 -> EC50)
3. **어류 (Fish):** 급성 독성 시험 (Probit 분석 -> LC50)
""")
st.divider()

# 사이드바에서 실험 종류 선택
analysis_type = st.sidebar.radio(
    "분석할 실험을 선택하세요",
    ["🟢 조류 성장저해 (ANOVA/NOEC)", "🦐 물벼룩 유영저해 (EC50)", "🐟 어류 급성독성 (LC50)"]
)

# -----------------------------------------------------------------------------
# [기능 1] 조류 성장저해 분석 함수 (세포수 입력 버전)
# -----------------------------------------------------------------------------
def run_algae_analysis():
    st.header("🟢 조류 성장저해 시험 분석")
    st.info("농도별 최종 **세포수(Cell Count)**를 입력하여 유의차(독성 여부)를 검정합니다.")

    # 1. 데이터 입력 초기값
    if 'algae_data' not in st.session_state:
        st.session_state.algae_data = pd.DataFrame({
            '농도(mg/L)': [0, 0, 0, 10, 10, 10, 32, 32, 32, 100, 100, 100],
            '세포수 (cells/mL)': [
                1000000, 1050000, 980000,  # 0 mg/L (대조군)
                950000, 920000, 940000,    # 10 mg/L
                700000, 680000, 720000,    # 32 mg/L
                300000, 280000, 310000     # 100 mg/L
            ]
        })

    # 2. 데이터 에디터
    df = st.data_editor(
        st.session_state.algae_data, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "세포수 (cells/mL)": st.column_config.NumberColumn(
                "세포수 (cells/mL)",
                format="%d"
            )
        }
    )

    if st.button("조류 통계 분석 실행"):
        col_name = '세포수 (cells/mL)'
        
        if df.empty or '농도(mg/L)' not in df.columns or col_name not in df.columns:
            st.error(f"데이터 형식을 확인해주세요. '{col_name}' 컬럼이 필요합니다.")
            return

        try:
            # 그룹화
            groups = df.groupby('농도(mg/L)')[col_name].apply(list)
            concentrations = sorted(groups.keys())
            control_group = groups[0] # 농도 0을 대조군으로 가정

            st.subheader("1. 기초 통계량")
            summary = df.groupby('농도(mg/L)')[col_name].agg(['mean', 'std', 'count']).reset_index()
            st.dataframe(summary.style.format("{:.2f}"))

            # --- 그래프 그리기 (Boxplot) ---
            st.subheader("📊 농도별 세포수 분포")
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_data = [groups[c] for c in concentrations]
            ax.boxplot(plot_data, labels=concentrations)
            ax.set_xlabel("Concentration (mg/L)")
            ax.set_ylabel("Cell Count (cells/mL)")
            ax.set_title("Cell Count by Concentration")
            st.pyplot(fig)

            # --- 정규성 검정 ---
            st.subheader("2. 정규성 검정 (Shapiro-Wilk)")
            normality_results = []
            for conc in concentrations:
                data = groups[conc]
                if len(data) >= 3:
                    stat, p = stats.shapiro(data)
                    normality_results.append({'농도': conc, 'P-value': f"{p:.4f}", '결과': '정규성 만족' if p > 0.05 else '정규성 위배'})
                else:
                    normality_results.append({'농도': conc, 'P-value': '-', '결과': '데이터 부족'})
            st.table(pd.DataFrame(normality_results))

            # --- 등분산성 검정 ---
            st.subheader("3. 등분산성 검정 (Levene)")
            data_list = [groups[conc] for conc in concentrations]
            l_stat, l_p = stats.levene(*data_list)
            st.write(f"P-value: **{l_p:.4f}** ({'등분산 만족' if l_p > 0.05 else '이분산(등분산 위배)'})")

            # --- ANOVA ---
            st.subheader("4. 통계적 유의성 검정 (One-way ANOVA)")
            f_stat, f_p = stats.f_oneway(*data_list)
            st.write(f"ANOVA P-value: **{f_p:.4f}**")

            if f_p < 0.05:
                st.success("통계적으로 유의한 차이가 있습니다. (P < 0.05)")
                
                # --- 사후 검정 ---
                st.subheader("5. NOEC / LOEC 도출 (Dunnett's type)")
                st.caption("대조군(0 mg/L)과 각 농도군 간의 비교 분석 결과입니다.")
                
                noec = 0
                loec = None
                comparisons = []
                alpha = 0.05 / (len(concentrations) - 1) 
                
                for conc in concentrations:
                    if conc == 0: continue
                    
                    equal_var_option = (l_p > 0.05)
                    t_stat, t_p = stats.ttest_ind(control_group, groups[conc], equal_var=equal_var_option)
                    
                    is_sig = t_p < alpha
                    # 들여쓰기 오류가 발생했던 부분 수정 완료
                    comparisons.append({
                        '비교 농도': conc, 
                        'T-stat': f"{t_stat:.2f}", 
                        'P-value': f"{t_p:.4f}", 
                        '보정된 Alpha': f"{alpha:.4f}",
                        '판정': '🚨 유의한 감소(독성)' if is_sig else '✅ 차이 없음'
                    })
                    
                    if is_sig and loec is None:
                        loec = conc
                    if not is_sig:
                        noec = conc
                
                st.table(pd.DataFrame(comparisons))
                
                c1, c2 = st.columns(2)
                c1.metric("NOEC (최대무영향농도)", f"{noec} mg/L")
                c2.metric("LOEC (최소영향농도)", f"{loec if loec else '> 최대농도'} mg/L")

            else:
                st.info("농도 간 유의한 차이가 없습니다. (P > 0.05)")
                st.write(f"NOEC: > {max(concentrations)} mg/L")

        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")


# -----------------------------------------------------------------------------
# [기능 2] 어류/물벼룩 Probit 분석 함수
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
# [메인] 선택에 따른 화면 표시
# -----------------------------------------------------------------------------
if "조류" in analysis_type:
    run_algae_analysis()
elif "물벼룩" in analysis_type:
    run_probit_analysis("🦐 물벼룩 급성 유영저해", "EC50")
elif "어류" in analysis_type:
    run_probit_analysis("🐟 어류 급성 독성", "LC50")
