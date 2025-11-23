import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생태독성 LC50/EC50 산출기", page_icon="🧪")

st.title("🧪 생태독성 데이터 분석 (Probit)")
st.markdown("""
이 어플리케이션은 **Probit 분석법(Log-Normal Model)**을 사용하여 
생태독성 실험의 **LC50 (반수치사농도)** 또는 **EC50 (반수영향농도)**를 산출합니다.
""")

st.divider()

# -----------------------------------------------------------------------------
# 2. 실험 종류 선택 및 데이터 입력 설정
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    test_type = st.radio(
        "🔬 실험 종류 선택",
        ('어류 급성독성 (LC50)', '물벼룩 유영저해 (EC50)')
    )

# 선택된 실험에 따라 라벨 텍스트 변경
if 'LC50' in test_type:
    value_label = "LC50"
    effect_label = "사망"
else:
    value_label = "EC50"
    effect_label = "유영저해"

# 초기 데이터셋 생성 (없을 경우에만)
if 'input_data' not in st.session_state:
    st.session_state.input_data = pd.DataFrame({
        '농도(mg/L)': [0.0, 6.25, 12.5, 25.0, 50.0, 100.0],
        '총 개체수': [10, 10, 10, 10, 10, 10],
        '반응 수': [0, 0, 1, 5, 9, 10]  # 사망하거나 유영저해된 수
    })

with col2:
    st.subheader("📊 데이터 입력")
    st.caption(f"각 농도별 총 개체수와 {effect_label} 개체수를 입력하세요.")
    
    # 데이터 에디터 (사용자가 수정 가능)
    edited_df = st.data_editor(
        st.session_state.input_data,
        num_rows="dynamic",
        use_container_width=True
    )

st.divider()

# -----------------------------------------------------------------------------
# 3. 계산 로직 및 결과 출력
# -----------------------------------------------------------------------------
if st.button("🚀 결과 계산하기"):
    try:
        # 1) 데이터 가져오기
        df = edited_df.copy()
        
        # 2) 유효성 검사
        # 농도가 0인 대조군은 로그 변환이 불가능하므로 회귀분석에서는 제외 (단, 0일 때 반응이 없다는 가정)
        df_calc = df[df['농도(mg/L)'] > 0].copy()

        # 데이터가 너무 적으면 계산 불가
        if len(df_calc) < 3:
            st.error("⚠️ 정확한 분석을 위해 최소 3개 이상의 농도 구간 데이터가 필요합니다.")
        else:
            # 3) Probit 변환 준비
            # 반응률 = 반응 수 / 총 개체수
            df_calc['반응률'] = df_calc['반응 수'] / df_calc['총 개체수']

            # 반응률이 0(0%)이거나 1(100%)이면 Probit 값이 무한대가 되므로 미세 보정
            # 통상적인 약식 계산에서는 0 -> 0.001, 1 -> 0.999 정도로 치환하여 계산
            df_calc['반응률_보정'] = df_calc['반응률'].clip(0.001, 0.999)

            # Probit 값 산출 (표준정규분포의 역함수, ppf)
            df_calc['Probit'] = stats.norm.ppf(df_calc['반응률_보정'])
            
            # 농도 로그 변환 (Log10)
            df_calc['Log_농도'] = np.log10(df_calc['농도(mg/L)'])

            # 4) 선형 회귀 분석 (Linear Regression)
            # X축: Log_농도, Y축: Probit
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_calc['Log_농도'], df_calc['Probit'])

            # 5) LC50 / EC50 산출
            # Probit 모델에서 반응률 50%는 Z값(Probit)이 0일 때입니다.
            # 식: 0 = slope * log(LC50) + intercept
            # 따라서 log(LC50) = -intercept / slope
            log_50 = -intercept / slope
            calculated_value = 10 ** log_50

            # ---------------- 결과 화면 표시 ----------------
            st.success("분석이 완료되었습니다!")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric(label=f"추정 {value_label} 값", value=f"{calculated_value:.2f} mg/L")
            with res_col2:
                st.metric(label="결정계수 (R²)", value=f"{r_value**2:.4f}")
            with res_col3:
                st.metric(label="기울기 (Slope)", value=f"{slope:.2f}")

            # 회귀식 보여주기
            st.info(f"📈 도출된 회귀식:  Y (Probit) = {slope:.4f} × log(농도) + ({intercept:.4f})")

            # ---------------- 그래프 그리기 ----------------
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # 실제 데이터 점 찍기
            ax.scatter(df_calc['Log_농도'], df_calc['Probit'], color='blue', label='Measured Data', zorder=5)
            
            # 회귀선 그리기
            x_min = df_calc['Log_농도'].min()
            x_max = df_calc['Log_농도'].max()
            # 그래프를 좀 더 길게 그려서 시각적으로 좋게 만듦
            x_range = np.linspace(x_min - 0.2, x_max + 0.2, 100)
            y_pred = slope * x_range + intercept
            
            ax.plot(x_range, y_pred, color='red', linestyle='-', label='Probit Regression Line')
            
            # 50% 지점 (Probit=0) 표시선
            ax.axhline(0, color='green', linestyle='--', alpha=0.5, label='50% Response (Probit=0)')
            ax.axvline(log_50, color='green', linestyle='--', alpha=0.5)
            
            # 그래프 꾸미기
            ax.set_xlabel('Log Concentration (log mg/L)')
            ax.set_ylabel('Probit (Standard Deviation Units)')
            ax.set_title(f'{test_type} Analysis Result')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend()
            
            # Streamlit에 그래프 출력
            st.pyplot(fig)

    except Exception as e:
        st.error("계산 중 오류가 발생했습니다.")
        st.write("에러 상세:", e)
        st.warning("입력 데이터에 문자가 있거나 빈 칸이 없는지 확인해주세요.")
