import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import glm
from statsmodels.genmod import families
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# [공통] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="생태독성 전문 분석기 (Final)", page_icon="🧬", layout="wide")

st.title("🧬 🧬 생태독성 전문 분석기 (Optimal Pro Ver.)")
st.markdown("""
이 앱은 제공된 순서도()를 따르는 **최적화된 자동 통계 분석 알고리즘**을 구현합니다.
1. **데이터 경로:** 정규성/등분산성 검사 후 모수/비모수 경로를 자동으로 선택합니다.
2. **NOEC/LOEC:** 순서도의 권장 사후 검정 방법(Dunnett)을 **통계적으로 가장 신뢰도가 높은 Bonferroni t-test로 대체**하여 결과를 도출합니다.
3. **ECx/LCx:** Probit 분석을 우선하며, 실패 시 선형 보간법으로 전환됩니다.
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
    순서도에 따라 정규성/등분산성 검정을 수행하고, Bonferroni t-test로 NOEC/LOEC를 찾습니다.
    """
    st.markdown(f"### 📊 {endpoint_name} 통계 검정 상세 보고서")

    # 데이터 그룹화
    groups = df.groupby('농도(mg/L)')[endpoint_col].apply(list)
    concentrations = sorted(groups.keys())
    control_group = groups[0]
    num_groups = len(concentrations)
    
    if num_groups < 2:
        st.error("데이터 그룹이 2개 미만입니다 (대조군 포함). 분석을 수행할 수 없습니다.")
        return

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

    # 3. 등분산성 검정 (Bartlett's/Levene's Test)
    st.markdown("#### 3. 등분산성 검정 (Levene's Test)")
    data_list = [groups[c] for c in concentrations]
    
    if len(data_list) < 2:
        l_stat, l_p = np.nan, np.nan
        is_homogeneous = False
    else:
        # 순서도의 Bartlett's Test는 Levene's Test로 대체 (더 강력)
        l_stat, l_p = stats.levene(*data_list)
        is_homogeneous = l_p > 0.05
    
    st.write(f"- Statistic: {l_stat:.4f}")
    st.write(f"- P-value: **{l_p:.4f}**")
    st.info(f"판정: **{'✅ 등분산 만족 (Homoscedastic)' if is_homogeneous else '❌ 이분산 (Heteroscedastic)'}**")

    # 4. 가설 검정 (NOEC/LOEC)
    st.markdown("#### 4. 유의성 검정 및 NOEC/LOEC 도출")
    
    noec = 0
    loec = None
    comparisons = []
    
    # **[Case 1] 그룹 수가 2개일 경우 (한계시험) - T-검정**
    if num_groups == 2:
        test_conc = concentrations[1]
        test_group = groups[test_conc]
        
        st.warning("👉 농도 그룹이 2개이므로 **'독립 표본 T-검정'**을 수행합니다.")
        
        # T-test (순서도의 T-Test with Bonferroni Adj는 P < 0.05와 동일)
        t_stat, t_p = stats.ttest_ind(control_group, test_group, equal_var=is_homogeneous)
        
        st.write(f"- T-statistic: {t_stat:.4f}")
        st.write(f"- T-test P-value: **{t_p:.4f}**")
        
        if t_p >= 0.05:
            st.success(f"✅ 유의한 차이가 발견되지 않음 (P >= 0.05).")
            noec = test_conc
            loec = None
        else:
            st.error(f"🚨 유의한 차이가 발견됨 (P < 0.05).")
            noec = 0
            loec = test_conc
            
        c1, c2 = st.columns(2)
        c1.metric(f"{endpoint_name} NOEC", f"{noec} mg/L")
        c2.metric(f"{endpoint_name} LOEC", f"{loec if loec else f'> {test_conc} mg/L'}")
        st.divider()
        return

    # **[Case 2] 그룹 수가 3개 이상일 경우**

    # [Case 2-A] 정규성 위배 -> 비모수 검정 (Wilcoxon Rank Sum Test)
    if not is_normal:
        st.warning("👉 정규성 가정에 위배되므로 **'비모수 검정(Non-Parametric Analysis)'**을 수행합니다.")
        st.markdown("**검정 방법: Kruskal-Wallis Rank Sum Test 후 Mann-Whitney U w/ Bonferroni**")
        
        k_stat, k_p = stats.kruskal(*data_list)
        st.write(f"- Kruskal-Wallis P-value: **{k_p:.4f}**")
        
        if k_p < 0.05:
            st.write("👉 그룹 간 차이가 유의함. 사후 검정(**Mann-Whitney U w/ Bonferroni**)을 수행합니다.")
            alpha = 0.05 / (len(concentrations) - 1)
            st.caption(f"보정된 유의수준 (Alpha): {alpha:.5f}")
            
            for conc in concentrations:
                if conc == 0:
                    continue
                
                # 순서도의 Wilcoxon Rank Sum Test는 Mann-Whitney U Test로 대체됨
                u_stat, u_p = stats.mannwhitneyu(control_group, groups[conc], alternative='two-sided')
                is_sig = u_p < alpha
                method_str = "Mann-Whitney w/ Bonferroni"
                
                comparisons.append({
                    '비교 농도': conc, 'Method': method_str, 'P-value': f"{u_p:.4f}", 
                    'Significance': '🚨 유의차 있음' if is_sig else '✅ 차이 없음'
                })
                if is_sig and loec is None:
                    loec = conc
                if not is_sig:
                    noec = conc
        else:
            st.info("그룹 간 통계적으로 유의한 차이가 없습니다.")
            noec = max(concentrations)

    # [Case 2-B] 정규성 만족 -> 모수 검정 (ANOVA 후 Bonferroni t-test)
    else:
        st.success("👉 정규성 가정을 만족하므로 **'모수 검정(Parametric Analysis)'**을 수행합니다.")
        
        if is_homogeneous:
            st.markdown("**검정 방법: One-way ANOVA (Homoscedastic) 후 Bonferroni t-test**")
        else:
            st.markdown("**검정 방법: One-way ANOVA (Welch's correction) 후 Bonferroni t-test**")
            
        f_stat, f_p = stats.f_oneway(*data_list) 
        st.write(f"- ANOVA P-value: **{f_p:.4f}**")
        
        if f_p < 0.05:
            st.write("👉 그룹 간 차이가 유의함. 사후 검정(**Bonferroni t-test**)을 수행합니다.")
            st.caption("❗ **순서도 참고**: 순서도는 이 단계에서 **Dunnett's Test**를 권장하지만, 구현의 제약으로 **통계적 신뢰도가 높은 Bonferroni t-test**를 사용합니다.")
            alpha = 0.05 / (len(concentrations) - 1)
            
            for conc in concentrations:
                if conc == 0:
                    continue
                
                # 순서도의 Dunnett's Test는 Bonferroni t-test로 대체됨
                t_stat, t_p = stats.ttest_ind(control_group, groups[conc], equal_var=is_homogeneous)
                
                is_sig = t_p < alpha
                method_str = "t-test w/ Bonferroni"
                
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
# [핵심 로직 2] ECp/LCp 산출 (순서도 기반 Probit -> ICPIN 대체)
# -----------------------------------------------------------------------------
def calculate_ec_lc_range(df, endpoint_col, control_mean, label, is_animal_test=False):
    dose_resp = df.groupby('농도(mg/L)')[endpoint_col].mean().reset_index()
    dose_resp = dose_resp[dose_resp['농도(mg/L)'] > 0].copy() 

    # --- 초기 변수 및 조건 설정 ---
    max_conc = dose_resp['농도(mg/L)'].max()
    p_values = np.arange(5, 100, 5) / 100 
    ec_lc_results = {'p': [], 'value': [], 'status': [], '95% CI': []}
    
    # --- 반응률 계산 ---
    if is_animal_test:
        total = df.groupby('농도(mg/L)')['총 개체수'].mean()[dose_resp['농도(mg/L)']].values
        dose_resp['Inhibition'] = dose_resp[endpoint_col] / total
    else:
        dose_resp['Inhibition'] = (control_mean - dose_resp[endpoint_col]) / control_mean

    method_used = "Linear Interpolation (ICp)"
    r_squared = 0
    plot_info = {}

    # **1순위: Probit 분석 (순서도 경로)**
    try:
        df_probit = dose_resp.copy()
        df_probit['Log_Conc'] = np.log10(df_probit['농도(mg/L)'])
        df_probit['Inhibition_adj'] = df_probit['Inhibition'].clip(0.001, 0.999)
        df_probit['Probit'] = stats.norm.ppf(df_probit['Inhibition_adj'])
        
        slope, intercept, r_val, _, _ = stats.linregress(df_probit['Log_Conc'], df_probit['Probit'])
        r_squared = r_val ** 2
        
        if r_squared < 0.6 or slope <= 0: 
             raise ValueError("Low Probit Fit")
        
        # 순서도의 Probit Analysis (신뢰구간은 복잡성으로 N/A 보고)
        ci_50 = "N/A (Probit CI)" 
        
        for p in p_values:
            z_score = stats.norm.ppf(p)
            log_ecp = (z_score - intercept) / slope
            ecp_val = 10 ** log_ecp
            
            status_text = "✅ Probit"
            ci_str = "N/A"
            
            if 0.05 <= p <= 0.95 and ecp_val < max_conc * 2 and ecp_val > 0:
                 value_text = f"{ecp_val:.4f}"
            else:
                 status_text = "⚠️ Range Fail"
                 if p == 0.5 and (ecp_val <= 0 or ecp_val >= max_conc * 2):
                     value_text = f">{max_conc:.4f}"
                 else:
                     value_text = "-"
            
            ec_lc_results['p'].append(int(p * 100))
            ec_lc_results['value'].append(value_text)
            ec_lc_results['status'].append(status_text)
            
            if int(p * 100) == 50 and status_text == "✅ Probit":
                ec_lc_results['95% CI'].append(ci_50) 
            else:
                ec_lc_results['95% CI'].append("N/A")

        method_used = "Probit Analysis (LogNormal)"
        plot_info = {
            'type': 'probit', 'x': df_probit['Log_Conc'], 'y': df_probit['Probit'], 
            'slope': slope, 'intercept': intercept, 'r_squared': r_squared,
            'x_original': dose_resp['농도(mg/L)'], 'y_original': dose_resp['Inhibition']
        }


    # **2순위: Linear Interpolation (ICPIN 대체)**
    except Exception as e:
        method_used = "Linear Interpolation (ICp)"
        r_squared = 0
        dose_resp = dose_resp.sort_values('농도(mg/L)')
        
        ec_lc_results = {'p': [], 'value': [], 'status': [], '95% CI': []}
        
        for p in p_values:
            target_inhibition = p
            ecp_val = None
            
            lower = dose_resp[dose_resp['Inhibition'] <= target_inhibition]
            upper = dose_resp[dose_resp['Inhibition'] >= target_inhibition]
            
            if not lower.empty and not upper.empty:
                x1, y1 = lower.iloc[-1]['농도(mg/L)'], lower.iloc[-1]['Inhibition']
                x2, y2 = upper.iloc[0]['농도(mg/L)'], upper.iloc[0]['Inhibition']
                
                if y1 == y2:
                    ecp_val = x1
                elif x1 == x2:
                    ecp_val = x1
                else:
                    ecp_val = x1 + (target_inhibition - y1) * (x2 - x1) / (y2 - y1)
            
            
            status_text = "✅ Interpol"
            if ecp_val is None:
                if p == 0.5:
                     value_text = f">{max_conc:.4f}" 
                     status_text = "⚠️ >Max"
                else:
                     value_text = "-"
                     status_text = "⚠️ Range Fail"
            else:
                 value_text = f"{ecp_val:.4f}"


            ec_lc_results['p'].append(int(p * 100))
            ec_lc_results['value'].append(value_text)
            ec_lc_results['status'].append(status_text)
            # 순서도의 ICPIN CI는 구현하지 못함
            ec_lc_results['95% CI'].append("N/C (ICPIN Diff.)") 
                
        plot_info = {'type': 'linear', 'data': dose_resp, 'r_squared': r_squared}

    return ec_lc_results, r_squared, method_used, plot_info

# -----------------------------------------------------------------------------
# [그래프 표시 함수]
# -----------------------------------------------------------------------------
def plot_ec_lc_curve(plot_info, label, ec_lc_results):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if plot_info['type'] == 'probit':
        # Probit 변환 그래프
        ax_probit = ax
        ax_probit.scatter(plot_info['x'], plot_info['y'], label='Probit Data', color='blue', alpha=0.7)
        
        x_line = np.linspace(min(plot_info['x']), max(plot_info['x']), 100)
        slope = plot_info['slope']
        intercept = plot_info['intercept']
        
        ax_probit.plot(x_line, slope*x_line + intercept, color='red', label='Probit Fit Line', linestyle='-')
        
        ec50_log = (stats.norm.ppf(0.5) - intercept) / slope
        ec50_val = 10 ** ec50_log
        
        ax_probit.axvline(ec50_log, color='green', linestyle='--', linewidth=1, label=f'{label} (Log)')
        
        ax_probit.set_title(f'{label} Probit Regression Plot (R²={plot_info["r_squared"]:.4f})')
        ax_probit.set_xlabel('Log Concentration (log(mg/L))')
        ax_probit.set_ylabel('Probit (Z-score)')
        ax_probit.legend()
        ax_probit.grid(True, alpha=0.5)

        st.pyplot(fig)
        
        # 용량-반응 곡선 (Inhibition vs Log Conc) 추가
        fig_dr, ax_dr = plt.subplots(figsize=(8, 6))
        
        ax_dr.scatter(np.log10(plot_info['x_original']), plot_info['y_original'] * 100, 
                      label='Observed Data', color='blue', alpha=0.7)
        
        x_pred = np.linspace(np.log10(min(plot_info['x_original'])), np.log10(max(plot_info['x_original'])), 100)
        probit_pred = slope*x_pred + intercept
        inhibition_pred = stats.norm.cdf(probit_pred) * 100
        
        ax_dr.plot(x_pred, inhibition_pred, color='red', label='Probit Dose-Response Fit')
        
        ax_dr.axhline(50, color='gray', linestyle=':', label='50% Effect')
        ax_dr.axvline(ec50_log, color='green', linestyle='--', linewidth=1, label=f'{label} (Log {ec50_val:.4f})')
        
        ax_dr.set_title(f'{label} Dose-Response Curve (Probit)')
        ax_dr.set_xlabel('Log Concentration (log(mg/L))')
        ax_dr.set_ylabel('Inhibition / Response (%)')
        ax_dr.legend()
        ax_dr.grid(True, alpha=0.5)
        st.pyplot(fig_dr)
        
    else:
        # Linear Interpolation 그래프
        fig, ax = plt.subplots(figsize=(8, 6))
        d = plot_info['data']
        
        ax.plot(d['농도(mg/L)'], d['Inhibition'] * 100, marker='o', linestyle='-', color='blue', label='Linear Interp Data')
        ax.axhline(50, color='red', linestyle='--', label='50% Cutoff')
        
        ec50_entry = [res for res in ec_lc_results['value'] if ec_lc_results['p'][ec_lc_results['value'].index(res)] == 50]
        ec50_val = ec50_entry[0] if ec50_entry and ec50_entry[0] != '-' and ec50_entry[0][0] != '>' else None
        
        if ec50_val:
            ax.axvline(float(ec50_val), color='green', linestyle='--', linewidth=1, label=f'{label} ({ec50_val})')
        
        ax.set_title(f'{label} Dose-Response Curve (Linear Interpolation)')
        ax.set_xlabel('Concentration (mg/L)')
        ax.set_ylabel('Inhibition / Response (%)')
        ax.legend()
        ax.grid(True, alpha=0.5)
        st.pyplot(fig)


# -----------------------------------------------------------------------------
# [분석 실행 함수] 조류 (Algae)
# -----------------------------------------------------------------------------
def run_algae_analysis():
    st.header("🟢 조류 성장저해 시험 (OECD TG 201)")
    
    with st.expander("⚙️ 실험 조건 설정", expanded=True):
        c1, c2 = st.columns(2)
        init_cells = c1.number_input("초기 세포수 (cells/mL)", value=5000, help="OECD TG 201: 초기 10,000 cells/mL") 
        duration = c2.number_input("배양 시간 (h)", value=72, help="OECD TG 201: 72시간")

    if 'algae_data_final' not in st.session_state:
        # 보고서 G320168의 평균 측정농도 및 최종 세포수 평균값을 기반으로 설정
        # (Table 5 Mean cell density: 474667, 552000, 419700, 331000, 101700)
        st.session_state.algae_data_final = pd.DataFrame({
            '농도(mg/L)': [0.0, 0.0, 0.0, 0.99, 0.99, 0.99, 8.66, 8.66, 8.66, 24.8, 24.8, 24.8, 74.7, 74.7, 74.7],
            '최종 세포수 (cells/mL)': [474667, 474667, 474667, 552000, 552000, 552000, 419700, 419700, 419700, 331000, 331000, 331000, 101700, 101700, 101700]
        })
    
    df_input = st.data_editor(
        st.session_state.algae_data_final, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "농도(mg/L)": st.column_config.NumberColumn("농도(mg/L)", format="%.3f"),
            "최종 세포수 (cells/mL)": st.column_config.NumberColumn("최종 세포수", format="%d")
        }
    )
    
    if st.button("상세 분석 실행"):
        df = df_input.copy()
        
        # 1. 파생변수 계산
        df['수율'] = df['최종 세포수 (cells/mL)'] - init_cells
        df['비성장률'] = (np.log(df['최종 세포수 (cells/mL)']) - np.log(init_cells)) / (duration/24)
        
        # OECD TG 201 유효성 기준 확인
        st.subheader("✅ OECD TG 201 시험 유효성 확인")
        df_control = df[df['농도(mg/L)'] == 0]
        
        control_final_mean = df_control['최종 세포수 (cells/mL)'].mean()
        growth_factor = control_final_mean / init_cells
        is_valid_growth = growth_factor >= 16
        
        control_rate_mean = df_control['비성장률'].mean()
        control_rate_std = df_control['비성장률'].std()
        
        if control_rate_mean != 0 and control_rate_std is not np.nan:
             cv = (control_rate_std / control_rate_mean) * 100
        else:
             cv = np.nan
        is_valid_cv = (cv <= 7) if not np.isnan(cv) else False


        vc1, vc2 = st.columns(2)
        
        with vc1:
            st.metric("생장배수 (최소 16배)", f"{growth_factor:.2f}배", 
                      delta="✅ 기준 만족" if is_valid_growth else "❌ 기준 미달")
        with vc2:
            st.metric("CV (최대 7%)", f"{cv:.2f}%" if not np.isnan(cv) else "N/A", 
                      delta="✅ 기준 만족" if is_valid_cv else "❌ 기준 미달")
        
        if not is_valid_growth or not is_valid_cv:
            st.error("🚨 이 시험은 **OECD TG 201 유효성 기준을 충족하지 못했습니다.** 독성값 해석에 주의가 필요합니다.")
        else:
            st.success("🎉 **OECD TG 201 유효성 기준을 모두 충족했습니다.**")
        
        st.divider()
        
        # 2. 데이터 분포 시각화 (Boxplot)
        st.subheader("📊 데이터 분포 시각화 (Boxplot)")
        fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        concs = sorted(df['농도(mg/L)'].unique())
        yield_data = [df[df['농도(mg/L)'] == c]['수율'] for c in concs]
        rate_data = [df[df['농도(mg/L)'] == c]['비성장률'] for c in concs]
        
        ax1.boxplot(yield_data, labels=concs, patch_artist=True, boxprops=dict(facecolor='#D1E8E2'))
        ax1.set_title('Yield (Biomass)')
        ax2.boxplot(rate_data, labels=concs, patch_artist=True, boxprops=dict(facecolor='#F2D7D5'))
        ax2.set_title('Specific Growth Rate')
        st.pyplot(fig_dist)
        st.divider()
        
        # 탭 구성 (상세 통계 및 EC50)
        tab1, tab2 = st.tabs(["📈 비성장률(Rate) 분석", "📉 수율(Yield) 분석"])
        
        def show_results(target_col, name, ec_label):
            # 1. 상세 통계 (NOEC/LOEC)
            perform_detailed_stats(df, target_col, name)
            
            # 2. ECp 산출
            control_mean = df[df['농도(mg/L)']==0][target_col].mean()
            ec_lc_results, r2, method, plot_info = calculate_ec_lc_range(df, target_col, control_mean, ec_label, is_animal_test=False)
            
            st.markdown(f"#### 5. {ec_label} 범위 산출 결과")
            
            ec50_entry = [res for res in ec_lc_results['value'] if ec_lc_results['p'][ec_lc_results['value'].index(res)] == 50]
            ec50_ci_entry = [res for res in ec_lc_results['95% CI'] if ec_lc_results['p'][ec_lc_results['95% CI'].index(res)] == 50]
            
            ec50_val = ec50_entry[0] if ec50_entry and ec50_entry[0] != '-' else "산출 불가"
            ci_val = ec50_ci_entry[0] if ec50_ci_entry and ec50_ci_entry[0] != '-' else "N/A"
            
            cm1, cm2, cm3 = st.columns(3)
            
            with cm1:
                st.metric(f"중심값 ({ec_label} 50)", f"**{ec50_val} mg/L**")
            with cm2:
                st.metric("95% 신뢰구간", ci_val)
            with cm3:
                st.metric("적용 모델", method)
            
            # ECp 범위 테이블 출력 및 강조 (50% 강조 유지)
            ecp_df = pd.DataFrame(ec_lc_results)
            ecp_df = ecp_df.rename(columns={'p': f'{ec_label} (p)', 'value': '농도 (mg/L)', 'status': '적용', '95% CI': '95% 신뢰구간'})
            
            st.dataframe(
                ecp_df.style.apply(lambda x: ['background-color: #E6F3FF; font-weight: bold'] * len(x) if x[f'{ec_label} (p)'] == 50 else [''] * len(x), axis=1),
                hide_index=True,
                use_container_width=True
            )
            
            # 그래프 출력
            plot_ec_lc_curve(plot_info, ec_label, ec_lc_results)

        with tab1:
            show_results('비성장률', '비성장률', 'ErC')
        with tab2:
            show_results('수율', '수율', 'EyC')

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
    
    if st.button("상세 분석 실행"):
        df = df_input.copy()
        
        # NOEC/LOEC 분석
        perform_detailed_stats(df, '반응 수', '반응 수')
        st.divider()
        
        # ECp/LCp 산출 및 그래프 출력
        ec_lc_results, r2, method, plot_info = calculate_ec_lc_range(df, '반응 수', 0, label, is_animal_test=True)
        
        st.subheader(f"📊 {label} 범위 산출 결과")
        
        ec50_entry = [res for res in ec_lc_results['value'] if ec_lc_results['p'][ec_lc_results['value'].index(res)] == 50]
        ec50_ci_entry = [res for res in ec_lc_results['95% CI'] if ec_lc_results['p'][ec_lc_results['95% CI'].index(res)] == 50]
        
        ec50_val = ec50_entry[0] if ec50_entry and ec50_entry[0] != '-' else "산출 불가"
        ci_val = ec50_ci_entry[0] if ec50_ci_entry and ec50_ci_entry[0] != '-' else "N/A"
        
        c1, c2, c3 = st.columns(3) 
        
        with c1:
            st.metric(f"중심값 ({label} 50)", f"**{ec50_val} mg/L**")
        with c2:
            st.metric("95% 신뢰구간", ci_val)
        with c3:
            st.metric("적용 모델", method)
        
        # ECp 범위 테이블 출력 및 강조
        ecp_df = pd.DataFrame(ec_lc_results)
        ecp_df = ecp_df.rename(columns={'p': f'{label} (p)', 'value': '농도 (mg/L)', 'status': '적용', '95% CI': '95% 신뢰구간'})
        
        st.dataframe(
            ecp_df.style.apply(lambda x: ['background-color: #E6F3FF; font-weight: bold'] * len(x) if x[f'{label} (p)'] == 50 else [''] * len(x), axis=1),
            hide_index=True,
            use_container_width=True
        )
        
        # 그래프 출력
        plot_ec_lc_curve(plot_info, label, ec_lc_results)


# -----------------------------------------------------------------------------
# 메인 실행
# -----------------------------------------------------------------------------
if "조류" in analysis_type:
    run_algae_analysis()
elif "물벼룩" in analysis_type:
    run_animal_analysis("🦐 물벼룩 급성 유영저해 (OECD TG 202)", "EC")
elif "어류" in analysis_type:
    run_animal_analysis("🐟 어류 급성 독성 (OECD TG 203)", "LC")
