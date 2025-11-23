import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# [기본 설정] 페이지 및 스타일
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생태독성 전문 분석기 (Final)", page_icon="🧬", layout="wide")

# 한글 폰트 설정 (OS 호환성 고려)
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['axes.unicode_minus'] = False

st.title("🧬 생태독성 전문 분석기 (OECD TG 통합 버전)")
st.markdown("""
이 앱은 **OECD Test Guidelines (TG 201, 202, 203)**에 준하는 분석을 수행합니다.
1. **데이터 입력:** 웹에서 직접 데이터 수정 가능
2. **통계 분석:** 정규성/등분산성 검정 후 NOEC/LOEC 도출 (모수/비모수 자동 선택)
3. **독성값 산출:** Hill Equation을 이용한 **EC5 ~ EC95 전 구간 산출 및 그래프**
""")
st.divider()

# -----------------------------------------------------------------------------
# [분석 모듈 1] 상세 통계 분석 (NOEC/LOEC)
# -----------------------------------------------------------------------------
def perform_detailed_stats(df, conc_col, response_col):
    st.markdown("### 📊 1. 통계적 가설 검정 (NOEC/LOEC)")
    
    # 데이터 그룹화
    groups = df.groupby(conc_col)[response_col].apply(list)
    concentrations = sorted(groups.keys())
    control_group = groups[concentrations[0]] # 농도 0 (Control)

    # (1) 정규성 검정
    is_normal = True
    norm_res = []
    for conc in concentrations:
        g_data = groups[conc]
        # 데이터가 3개 이상이고 분산이 0이 아닐 때만 검정 수행
        if len(g_data) >= 3 and np.std(g_data) > 0:
            s, p = stats.shapiro(g_data)
            res = 'Normal' if p > 0.01 else 'Non-Normal'
            norm_res.append({'Conc': conc, 'P-value': f"{p:.4f}", 'Result': res})
            if p <= 0.01: is_normal = False
        else:
            norm_res.append({'Conc': conc, 'P-value': '-', 'Result': 'Skip (N<3 or Var=0)'})
    
    # (2) 등분산성 검정
    data_list = [groups[c] for c in concentrations]
    if len(data_list) > 1:
        # 모든 그룹의 분산이 0인 경우(완벽히 같은 값) Levene 검정 오류 방지
        try:
            l_stat, l_p = stats.levene(*data_list)
            is_homogeneous = l_p > 0.05
        except:
            l_p = 1.0
            is_homogeneous = True
    else:
        l_p = 0.0
        is_homogeneous = False

    # (3) 결과 요약 출력
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### 정규성 (Shapiro-Wilk)")
        st.dataframe(pd.DataFrame(norm_res), use_container_width=True)
    with col2:
        st.write("#### 등분산성 & 분석 방법")
        st.write(f"- Levene P-value: **{l_p:.4f}** ({'등분산' if is_homogeneous else '이분산'})")
        
        test_type = "param"
        if not is_normal:
            st.warning("👉 **비모수 검정 (Kruskal-Wallis)** 채택")
            test_type = "non-param"
        else:
            st.success("👉 **모수 검정 (ANOVA)** 채택")
            test_type = "param"

    # (4) 유의성 검정 (Control vs Treatment)
    st.write("#### 사후 검정 결과")
    comparisons = []
    noec, loec = max(concentrations), None 

    # Bonferroni 보정 (다중비교)
    alpha = 0.05 / (len(concentrations) - 1) if len(concentrations) > 1 else 0.05

    for conc in concentrations:
        if conc == concentrations[0]: continue # Control 제외
        
        is_sig = False
        p_val = 1.0
        method = ""

        try:
            if test_type == "non-param":
                # Mann-Whitney U
                u, p_val = stats.mannwhitneyu(control_group, groups[conc], alternative='two-sided')
                method = "Mann-Whitney"
            else:
                # T-test (Welch or Student)
                t, p_val = stats.ttest_ind(control_group, groups[conc], equal_var=is_homogeneous)
                method = "Welch's t-test" if not is_homogeneous else "t-test"
        except:
            p_val = 1.0
            method = "Error"

        is_sig = p_val < alpha
        
        comparisons.append({
            '비교 농도': conc, 
            'Method': method, 
            'P-value': f"{p_val:.4f}", 
            'Significance': '🚨 유의차 있음 (LOEC)' if is_sig else '✅ 차이 없음'
        })

        if is_sig:
            if loec is None: loec = conc 
        else:
            if loec is None: noec = conc

    st.dataframe(pd.DataFrame(comparisons), use_container_width=True)
    st.info(f"📍 **결론: NOEC = {noec} mg/L, LOEC = {loec if loec else '> ' + str(max(concentrations))} mg/L**")


# -----------------------------------------------------------------------------
# [분석 모듈 2] 용량-반응 곡선 및 ECx/LCx (Hill Equation)
# -----------------------------------------------------------------------------
def hill_equation(x, top, bottom, ec50, hill_slope):
    """4-Parameter Logistic Equation"""
    return bottom + (top - bottom) / (1 + (x / ec50)**(-hill_slope))

def inverse_hill(y, top, bottom, ec50, hill_slope):
    """Hill 식의 역함수 (y -> x 계산)"""
    if hill_slope > 0: # 증가 함수 (독성 반응)
        if y >= top: return np.inf
        if y <= bottom: return 0
    else: # 감소 함수 (성장률 등)
        if y <= top: return np.inf
        if y >= bottom: return 0
        
    try:
        return ec50 * (( (top - bottom) / (y - bottom) ) - 1)**(1 / -hill_slope)
    except:
        return 0

def calculate_dose_response(df, conc_col, response_col, label="Response"):
    st.markdown("### 📈 2. 농도-반응 곡선 및 ECx/LCx 산출")
    
    x_data = df[conc_col].values
    y_data = df[response_col].values
    
    # 0농도는 로그 스케일 그래프를 위해 아주 작은 값으로 대체 (피팅엔 영향 없음)
    x_fit = x_data.copy()
    x_fit[x_fit == 0] = 1e-9

    # 초기 추정값 설정 (자동화)
    # 일반적인 독성 반응: 농도 증가 -> 반응값(치사율/저해율) 증가 (Slope > 0)
    # Top=100, Bottom=0 가정
    p0 = [100, 0, np.median(x_data[x_data > 0]) if len(x_data[x_data>0])>0 else 10, 2]
    
    try:
        # Curve Fitting
        popt, pcov = curve_fit(hill_equation, x_fit, y_data, p0=p0, maxfev=5000)
        top_fit, bot_fit, ec50_fit, slope_fit = popt
        
        st.success(f"✅ 모델 피팅 성공! (EC50 ≈ {ec50_fit:.4f})")
        
        col_g, col_t = st.columns([1.5, 1])
        
        with col_g:
            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x_data, y_data, color='black', alpha=0.6, label='Observed Data', zorder=5)
            
            # 곡선 그리기
            x_min = min(x_data[x_data > 0]) if len(x_data[x_data > 0]) > 0 else 0.01
            x_curve = np.logspace(np.log10(x_min*0.1), np.log10(max(x_data)*1.5), 200)
            y_curve = hill_equation(x_curve, *popt)
            
            ax.plot(x_curve, y_curve, color='blue', linewidth=2, label='Fitted Curve')
            
            # EC50 표시
            ax.axhline(50, color='red', linestyle='--', alpha=0.5)
            ax.axvline(ec50_fit, color='red', linestyle='--', alpha=0.5, label=f'EC50')

            ax.set_xscale('log')
            ax.set_xlabel(f"{conc_col} (Log Scale)", fontsize=10)
            ax.set_ylabel(f"{label} (%)", fontsize=10)
            ax.set_title("Dose-Response Curve (OECD TG)", fontsize=12)
            ax.set_ylim(-10, 110)
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.legend()
            st.pyplot(fig)

        with col_t:
            # EC5 ~ EC95 테이블 산출
            st.write("#### 📋 독성값 산출표 (EC5 ~ EC95)")
            ec_results = []
            for level in range(5, 100, 5):
                calc_conc = inverse_hill(level, top_fit, bot_fit, ec50_fit, slope_fit)
                ec_results.append({
                    'Level': f"EC{level} / LC{level}",
                    'Effect(%)': level,
                    'Calc. Conc': calc_conc
                })
            
            res_df = pd.DataFrame(ec_results)
            # EC50 부분 하이라이트
            st.dataframe(
                res_df.style.highlight_between(left=49, right=51, axis=1, props='font-weight:bold; background-color:#ffffcc;')
                .format({"Calc. Conc": "{:.4f}"}),
                use_container_width=True,
                height=400
            )
        
    except Exception as e:
        st.error(f"곡선 피팅 실패: {e}")
        st.write("데이터의 경향성이 뚜렷하지 않거나 점의 개수가 부족할 수 있습니다.")


# -----------------------------------------------------------------------------
# [메인 실행부] 사용자 입력 및 로직 연결
# -----------------------------------------------------------------------------
analysis_type = st.sidebar.radio(
    "실험 종류 선택",
    ["🟢 조류 성장저해 (Algae)", "🦐 물벼룩 유영저해 (Daphnia)", "🐟 어류 급성독성 (Fish)"]
)

# 데이터 초기화 및 에디터 설정
st.subheader(f"📝 데이터 입력: {analysis_type}")

if "data" not in st.session_state:
    st.session_state.data = {}

# 실험별 기본 데이터 템플릿 제공
if analysis_type == "🟢 조류 성장저해 (Algae)":
    default_df = pd.DataFrame({
        '농도(mg/L)': [0]*3 + [10]*3 + [32]*3 + [100]*3,
        '최종 세포수': [100, 98, 102, 90, 88, 92, 50, 48, 52, 10, 12, 8]
    })
    conc_col = '농도(mg/L)'
    input_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)
    
    # 전처리: 저해율 계산 (Control 평균 대비)
    ctrl_mean = input_df[input_df[conc_col] == 0]['최종 세포수'].mean()
    input_df['Inhibition(%)'] = (ctrl_mean - input_df['최종 세포수']) / ctrl_mean * 100
    target_col = 'Inhibition(%)' # 분석 대상 컬럼

elif analysis_type == "🦐 물벼룩 유영저해 (Daphnia)":
    default_df = pd.DataFrame({
        '농도(mg/L)': [0]*4 + [6.25]*4 + [12.5]*4 + [25]*4 + [50]*4 + [100]*4,
        '유영저해율(%)': [0, 0, 0, 0,  5, 0, 5, 0,  20, 25, 20, 15,  80, 85, 90, 80,  100, 100, 100, 100,  100, 100, 100, 100]
    })
    conc_col = '농도(mg/L)'
    target_col = '유영저해율(%)'
    input_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)

else: # 어류
    default_df = pd.DataFrame({
        '농도(mg/L)': [0]*3 + [10]*3 + [100]*3,
        '치사율(%)': [0, 0, 0,  50, 40, 60,  100, 100, 100]
    })
    conc_col = '농도(mg/L)'
    target_col = '치사율(%)'
    input_df = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)

# 실행 버튼
if st.button("🚀 분석 실행"):
    st.divider()
    
    # 분석 탭 구성
    tab1, tab2 = st.tabs(["📊 통계 분석 (NOEC/LOEC)", "📈 독성값 산출 (EC5~95)"])
    
    with tab1:
        perform_detailed_stats(input_df, conc_col, target_col)
        
    with tab2:
        calculate_dose_response(input_df, conc_col, target_col, label=target_col)

