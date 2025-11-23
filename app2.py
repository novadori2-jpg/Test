import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 페이지 설정
st.set_page_config(page_title="생태독성 LC50/EC50 계산기", page_icon="🧪")

st.title("🧪 생태독성 LC50 / EC50 산출기")
st.write("데이터를 입력하면 **Probit 분석법**을 통해 반수치사농도(LC50) 또는 반수영향농도(EC50)를 계산합니다.")

# 1. 실험 종류 선택 (사이드바 혹은 메인화면)
test_type = st.radio(
    "실험 종류를 선택하세요:",
    ('어류 급성독성 (LC50)', '물벼룩 유영저해 (EC50)')
)

# 변수명 설정 (LC50이냐 EC50이냐에 따라 라벨 변경)
result_label = "LC50" if "LC50" in test_type else "EC50"
effect_label = "사망" if "LC50" in test_type else "유영저해"

st.divider()

# 2. 데이터 입력
st.subheader(f"📊 {test_type} 데이터 입력")
st.info(f"농도별 총 개체수와 {effect_label} 개체수를 입력해주세요.")

# 초기 데이터셋 (예시 데이터)
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(
        {
            '농도(mg/L)': [0, 6.25, 12.5, 25.0, 50.0, 100.0],
            '총 개체수': [10, 10, 10, 10, 10, 10],
            '반응 수': [0, 0, 1, 5, 9, 10]
        }
    )

# 데이터 에디터 (사용자가 직접 수정 가능)
edited_df = st.data_editor(
    st.session_state.data,
    num_rows="dynamic",
    use_container_width=True
)

st.divider()

# 3. 계산 및 시각화
if st.button("결과 계산하기"):
    try:
        # 데이터 전처리
        df = edited_df.copy()
        
        # 유효성 검사: 농도가 0보다 큰 데이터만 사용 (로그 변환 위해 대조군 제외)
        df_calc = df[df['농도(mg/L)'] > 0].copy()
        
        if len(df_calc) < 3:
            st.warning("정확한 계산을 위해 최소 3개 이상의 농도 구간 데이터가 필요합니다.")
        else:
            # 반응률 계산
            df_calc['반응률'] = df_calc['반응 수'] / df_calc['총 개체수']
            
            # Probit 변환을 위한 보정 (0% -> 0.001, 100% -> 0.999)
            # 0이나 1은 Probit 변환 시 무한대가 되므로 미세하게 조정
            df_calc['반응률_보정'] = df_calc['반응률'].clip(0.001, 0.999)
            
            # Probit 변환 (표준정규분포의 역함수)
            df_calc['Probit'] = stats.norm.ppf(df_calc['반응률_보정'])
            
            # 로그 농도 (Log10)
            df_calc['Log_농도'] = np.log10(df_calc['농도(mg/L)'])

            # 선형 회귀 분석 (X: Log농도, Y: Probit)
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_calc['Log_농도'], df_calc['Probit'])

            # 결과 산출 (Probit = 0 일 때가 50% 반응)
            # 식: 0 = slope * log(Val) + intercept
            # log(Val) = -intercept / slope
            log_50 = -intercept / slope
            calculated_value = 10 ** log_50

            # --- 결과 출력 화면 ---
            st.subheader("📝 분석 결과")
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**{result_label} 값**")
                st.markdown(f"### {calculated_value:.4f} mg/L")
            
            with col2:
                st.info("**결정계수 ($R^2$)**")
                st.markdown(f"### {r_value**2:.4f}")
                st.caption("1에 가까울수록 회귀식의 신뢰도가 높습니다.")

            st.write(f"**회귀식:** $Y (Probit) = {slope:.4f} \\times \\log(X) + ({intercept:.4f})$")

            # --- 그래프 그리기 ---
            st.subheader("📈 Probit 회귀 곡선")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # 1. 실험 데이터 점 찍기
            ax.scatter(df_calc['Log_농도'], df_calc['Probit'], color='blue', label='Measured Data', zorder=3)
            
            # 2. 회귀선 그리기
            x_min = df_calc['Log_농도'].min()
            x_max = df_calc['Log_농도'].max()
            x_range = np.linspace(x_min - 0.2, x_max + 0.2, 100)
            y_pred = slope * x_range + intercept
            
            ax.plot(x_range, y_pred, color='red', linestyle='-', label='Regression Line')
            
            # 3. 50% 지점 (Probit=0) 표시
            ax.axhline(0, color='green', linestyle='--', alpha=0.7, label='50% Response Level')
            ax.axvline(log_50, color='green', linestyle='--', alpha=0.7)
            
            # 그래프 꾸미기
            ax.set_xlabel('Log Concentration')
            ax.set_ylabel('Probit Unit')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            ax.set_title(f'{test_type} Probit Analysis')
            
            # Streamlit에 그래프 표시
            st.pyplot(fig)

    except Exception as e:
        st.error("계산 중 오류가 발생했습니다.")
        st.write(f"에러 내용: {e}")
        st.warning("데이터에 빈 칸이 있거나 숫자가 아닌 값이 있는지 확인해주세요.")

    f_val = anova_table['F'].iloc[0]; p_anova = anova_table['PR(>F)'].iloc[0]
    try: _, p_shapiro = stats.shapiro(model.resid)
    except: p_shapiro = 1.0
    groups = [df_clean[df_clean['Concentration'] == c][endpoint].values for c in concs]
    try: _, p_bartlett = stats.bartlett(*groups)
    except: p_bartlett = 1.0
    residuals = model.resid.values
    g_stat, g_crit, g_dec = calc_grubbs_test(residuals)
    mk_stat, mk_p, mk_dec = calc_mann_kendall(control_vals)
    auxiliary = {"grubbs": {"stat": g_stat, "crit": g_crit, "decision": g_dec}, "mk": {"stat": mk_stat, "p": mk_p, "decision": mk_dec}}
    assumptions = {"shapiro": p_shapiro, "bartlett": p_bartlett}
    is_parametric = (p_shapiro > 0.01) and (p_bartlett > 0.01)
    comparison_res = []
    noec, loec = concs[-1], "> Max"
    loec_found = False
    if is_parametric:
        method_name = "Dunnett Multiple Comparison Test"
        n_control = len(control_vals); crit_val = 2.506 
        for conc in concs[1:]:
            treat_vals = df_clean[df_clean['Concentration'] == conc][endpoint].values
            n_treat = len(treat_vals)
            mean_c, mean_t = np.mean(control_vals), np.mean(treat_vals)
            se = np.sqrt(ms_err * (1/n_control + 1/n_treat))
            t_stat = (mean_c - mean_t) / se if se > 0 else 0
            if t_stat < 0: t_stat = 0
            is_sig = t_stat > crit_val; p_val = stats.t.sf(t_stat, df_err)
            if is_sig and not loec_found:
                loec = conc; idx = concs.index(conc)
                noec = f"< {conc}" if idx == 1 else concs[idx-1]
                loec_found = True
            comparison_res.append({"conc": conc, "stat": t_stat, "crit": crit_val, "msd": se*crit_val, "p": p_val, "sig": "Significant Effect" if is_sig else "Non-Significant Effect"})
    else:
        method_name = "Wilcoxon Rank Sum Test (Bonferroni Adj.)"
        k = len(concs) - 1; alpha_adj = 0.05 / k
        for conc in concs[1:]:
            treat_vals = df_clean[df_clean['Concentration'] == conc][endpoint].values
            u_stat, p_val_raw = stats.mannwhitneyu(control_vals, treat_vals, alternative='greater')
            n1, n2 = len(control_vals), len(treat_vals)
            mu_u, sigma_u = n1*n2/2, np.sqrt(n1*n2*(n1+n2+1)/12)
            z_score = (u_stat - mu_u)/sigma_u if sigma_u > 0 else 0
            is_sig = p_val_raw < alpha_adj
            if is_sig and not loec_found:
                loec = conc; idx = concs.index(conc)
                noec = f"< {conc}" if idx == 1 else concs[idx-1]
                loec_found = True
            comparison_res.append({"conc": conc, "stat": z_score, "crit": "-", "msd": "-", "p": p_val_raw, "sig": "Significant Effect" if is_sig else "Non-Significant Effect"})
    if not loec_found: noec, loec = concs[-1], f"> {concs[-1]}"
    return {"method": method_name, "is_parametric": is_parametric, "anova": {"ss_bet": ss_bet, "df_bet": df_bet, "ms_bet": ms_bet, "f": f_val, "p": p_anova, "ss_err": ss_err, "df_err": df_err, "ms_err": ms_err} if is_parametric else None, "comparison": comparison_res, "auxiliary": auxiliary, "assumptions": assumptions, "noec": noec, "loec": loec}

# ==============================================================================
# 2. 모바일 전용 HTML 리포트 생성기 (단순화된 버전)
# ==============================================================================

def generate_mobile_report(meta, res_mu, res_yield, ec_mu, ec_yield):
    """모바일 화면에서 보기 편한 카드 형태의 HTML 리포트"""
    
    style = """
    <style>
        .card { background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
        .card-title { font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 10px; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
        .stat-row { display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 14px; }
        .stat-label { font-weight: bold; color: #555; }
        .stat-val { font-weight: bold; color: #e74c3c; }
        .table-mobile { width: 100%; font-size: 12px; border-collapse: collapse; }
        .table-mobile th { background-color: #eee; padding: 5px; text-align: left; border-bottom: 1px solid #ccc; }
        .table-mobile td { padding: 5px; border-bottom: 1px solid #eee; }
    </style>
    """
    
    html = f"{style}<h3>📊 분석 결과 리포트</h3>"
    
    # 1. 요약 카드
    html += f"""
    <div class="card">
        <div class="card-title">📌 결론 요약 (Summary)</div>
        <div class="stat-row"><span class="stat-label">Specific Growth Rate NOEC:</span><span class="stat-val">{res_mu['noec']} mg/L</span></div>
        <div class="stat-row"><span class="stat-label">Specific Growth Rate LOEC:</span><span class="stat-val">{res_mu['loec']} mg/L</span></div>
        <hr>
        <div class="stat-row"><span class="stat-label">Yield NOEC:</span><span class="stat-val">{res_yield['noec']} mg/L</span></div>
        <div class="stat-row"><span class="stat-label">Yield LOEC:</span><span class="stat-val">{res_yield['loec']} mg/L</span></div>
    </div>
    """
    
    # 2. EC값 카드
    def ec_table(ec_data):
        t = '<table class="table-mobile"><tr><th>Level</th><th>mg/L</th><th>95% CI</th></tr>'
        for k in ['EC10', 'EC50', 'EC90']: # 모바일은 주요 값만 표시
            v = ec_data.get(k, {'val':'-'})
            ci = f"{v.get('lcl','-')}~{v.get('ucl','-')}"
            t += f"<tr><td>{k}</td><td>{v['val']}</td><td>{ci}</td></tr>"
        return t + "</table>"

    html += f"""
    <div class="card">
        <div class="card-title">📈 유효 농도 (EC Values)</div>
        <p><b>Specific Growth Rate</b></p>
        {ec_table(ec_mu)}
        <p style="margin-top:10px;"><b>Yield</b></p>
        {ec_table(ec_yield)}
    </div>
    """
    
    # 3. 상세 통계 (접이식 아님, 모바일은 스크롤)
    html += f"""
    <div class="card">
        <div class="card-title">🔍 통계 세부 정보</div>
        <p><b>Test Method:</b> {res_mu['method']}</p>
        <p><b>Shapiro-Wilk (Normality):</b> P={res_mu['assumptions']['shapiro']:.4f}</p>
        <p><b>Bartlett (Variance):</b> P={res_mu['assumptions']['bartlett']:.4f}</p>
    </div>
    """
    
    return html

# ==============================================================================
# 3. 모바일 앱 메인 로직 (Streamlit UI)
# ==============================================================================

def main():
    # 모바일 최적화 설정
    st.set_page_config(page_title="CETIS Mobile", layout="centered", initial_sidebar_state="collapsed")
    
    # CSS로 버튼과 폰트 크기 키우기 (터치 최적화)
    st.markdown("""
    <style>
        div.stButton > button { width: 100%; padding: 15px; font-size: 18px; font-weight: bold; border-radius: 10px; }
        div[data-testid="stExpander"] { border-radius: 10px; }
        input { font-size: 16px !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📱 CETIS Mobile Analysis")
    st.caption("모바일 환경에서 간편하게 독성 데이터를 분석하세요.")

    # --- [Tab 1] 데이터 입력 ---
    tab1, tab2 = st.tabs(["📝 데이터 입력", "📊 분석 결과"])
    
    with tab1:
        with st.expander("1. 실험 정보 설정", expanded=True):
            col1, col2 = st.columns(2)
            batch_id = col1.text_input("Batch ID", "BATCH-001")
            sample_id = col2.text_input("Sample ID", "SMP-01")
            control_name = st.text_input("Control Name", "Dilution Water")
            
        st.markdown("### 2. 측정값 입력")
        st.info("농도별로 **72시간 세포수(Final)**를 입력해주세요.")
        
        # 모바일 친화적 입력 폼 (Session State 활용)
        if 'input_data' not in st.session_state:
            # 기본 템플릿
            default_data = []
            # Control
            for i in range(6): default_data.append({"Group": "Control", "Conc": 0.0, "Rep": i+1, "Final": 480000})
            # Treats
            concs = [10.0, 17.0, 31.0, 56.0, 100.0]
            for c in concs:
                for i in range(3): default_data.append({"Group": "Treat", "Conc": c, "Rep": i+1, "Final": 450000})
            st.session_state.input_data = pd.DataFrame(default_data)

        # 초기값 입력 (상단 고정)
        initial_val = st.number_input("초기 세포수 (0h Initial)", value=10000, step=1000)

        # 데이터 에디터 (모바일에서 터치로 수정 가능)
        edited_df = st.data_editor(
            st.session_state.input_data,
            column_config={
                "Group": st.column_config.TextColumn("그룹", disabled=True, width="small"),
                "Conc": st.column_config.NumberColumn("농도(mg/L)", format="%.1f", width="small"),
                "Rep": st.column_config.NumberColumn("반복", disabled=True, width="small"),
                "Final": st.column_config.NumberColumn("72h 세포수", format="%d", width="medium")
            },
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic"
        )
        
        if st.button("🚀 분석 실행 (Analyze)", type="primary"):
            st.session_state.run_analysis = True
            st.session_state.final_df = edited_df.copy()
            st.session_state.final_df['Initial'] = initial_val
            st.session_state.final_df = st.session_state.final_df.rename(columns={"Conc": "Concentration"})
            st.rerun() # 결과 탭으로 이동하기 위해 리프레시

    # --- [Tab 2] 분석 결과 ---
    with tab2:
        if 'run_analysis' in st.session_state and st.session_state.run_analysis:
            try:
                df = st.session_state.final_df
                
                with st.spinner("통계 분석 중... (ICPIN + Dunnett/Wilcoxon)"):
                    # 1. 계산
                    df = calculate_growth_yield(df)
                    
                    # 2. 통계 (Mu & Yield)
                    det_mu = run_cetis_algorithm(df, 'Mu')
                    det_yield = run_cetis_algorithm(df, 'Yield')
                    ec_mu = get_icpin_values_with_ci(df, 'Mu')
                    ec_yield = get_icpin_values_with_ci(df, 'Yield')
                    
                    # 3. 요약 통계
                    def get_summ(d, c):
                        s = d.groupby('Concentration')[c].agg(['count','mean','std','min','max']).reset_index()
                        s.columns = ['Concentration','Count','Mean','StdDev','Min','Max']
                        c0 = s.loc[s['Concentration']==0, 'Mean'].values[0]
                        s['CV'] = (s['StdDev']/s['Mean'])*100
                        s['Effect'] = (1 - s['Mean']/c0)*100
                        return s
                    
                    summ_mu = get_summ(df, 'Mu')
                    summ_yield = get_summ(df, 'Yield')

                    # 4. 결과 표시 (모바일 HTML 리포트)
                    meta = {"batch_id": batch_id, "sample_id": sample_id, "control_name": control_name}
                    html_report = generate_mobile_report(meta, det_mu, det_yield, ec_mu, ec_yield)
                    
                    st.components.v1.html(html_report, height=600, scrolling=True)
                    
                    # 5. 전체 PDF 스타일 리포트 다운로드
                    # (이전 generate_annex6_html 함수는 너무 길어서 여기서는 생략했지만, 실제 앱에는 포함해서 다운로드 제공 가능)
                    st.download_button("📥 전체 리포트 다운로드 (HTML)", html_report, file_name="report.html", mime="text/html")
                    
            except Exception as e:
                st.error(f"분석 오류: {e}")
        else:
            st.info("👈 '데이터 입력' 탭에서 데이터를 넣고 [분석 실행] 버튼을 눌러주세요.")

if __name__ == "__main__":
    main()
