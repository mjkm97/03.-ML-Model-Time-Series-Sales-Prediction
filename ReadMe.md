# 프로젝트 3. 드럭스토어 판매액 예측모델

<aside>

### 목차

1. **서론:**  **프로젝트 배경 및 목적**
    
    **1-1. 데이터 선정 및 수집** 
    
    **1-2. 데이터 전처리**
    
    **1-3. EDA 를 통한 가설 설정 및 문제정의**
    

1. **본론:**
    
    **2-1. 예측 모델 구현을 위한 학습 진행 및 결과 확인**
    
    - Linear Logistics - Lasso, Ridge
    - Decision Tree
    - Random Forest
    - XGboost

1. **결론: 모델 성능 비교 및 해석 & 최종 모델 선정**
    
    **3-1. 구현된 매출 예측 모델 결과**
    
    **3-2. 최종 모델 선정 : XGboost**
    
    **3-3. XGBoost 기반 향후 6주간 매출 예측**
    
2. 역할분담
    - 공통: 전처리 로직 및 코드 리뷰
    - 김민주: EDA 및 회귀모델 및 보고서 총괄
    - 이유리 : Decision Tree
    - 임동현 : Random Forest, 기초 전처리
    - 임강민 : XGBoost
      
    
</aside>

## **프로젝트 배경 및 목적**

Rossmann 드럭스토어 체인의 데이터를 바탕으로, 다양한 매장 특성과 마케팅 요인이 매출에 어떤 영향을 미치는지 분석하고, 이를 기반으로 정확한 매출 예측 모델을 구축하는 것이 목적입니다.

## 1-1. 데이터 선정 및 수집

[Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)



Rossmann은 7개의 유럽 국가에서 3000개 이상의 드럭스토어를 운영하고 있습니다. 매출은 프로모션, 경쟁사, 학교 및 공휴일 등 다양한 요인에 영향을 받습니다. 이를 예측하기 위해 아래와 같은 데이터들을 모아놓은 데이터셋입니다.

- 데이터 칼럼 항목 및 정보
    
    ## 📁 **1. `train.csv`**
    
    > 매장별 일별 매출 및 관련 정보 (학습용 데이터)
    > 
    
    | 컬럼명 | 설명 |
    | --- | --- |
    | `Store` | 매장 ID |
    | `DayOfWeek` | 요일 (1=월요일, 7=일요일) |
    | `Date` | 날짜 (YYYY-MM-DD) |
    | `Sales` | 매출 (타깃 변수) |
    | `Customers` | 방문 고객 수 |
    | `Open` | 매장 개점 여부 (1=열림, 0=닫힘) |
    | `Promo` | 프로모션 진행 여부 (1=진행 중) |
    | `StateHoliday` | 주 공휴일 여부 (`0`=아님, `a`=공휴일, `b`=부활절, `c`=크리스마스 등) |
    | `SchoolHoliday` | 학교 방학 여부 (1=방학) |
    
    ---
    
    ## 📁 **2. `test.csv`**
    
    > 예측 대상 데이터 (Sales 없음)
    > 
    
    ---
    
    ## 📁 **3. `store.csv`**
    
    > 각 매장에 대한 고정 정보
    > 
    > 
    > 
    > | 컬럼명 | 설명 |
    > | --- | --- |
    > | `Store` | 매장 ID |
    > | `StoreType` | 매장 유형 (`a`, `b`, `c`, `d`) |
    > | `Assortment` | 상품 구성 수준 (`a`=기본, `b`=중간, `c`=광범위) |
    > | `CompetitionDistance` | 가장 가까운 경쟁 매장 거리 (미터) |
    > | `CompetitionOpenSinceMonth` | 경쟁 매장 개점 월 |
    > | `CompetitionOpenSinceYear` | 경쟁 매장 개점 연도 |
    > | `Promo2` | 연중 반복되는 프로모션 여부 (0 or 1) |
    > | `Promo2SinceWeek` | Promo2 시작 주차 |
    > | `Promo2SinceYear` | Promo2 시작 연도 |
    > | `PromoInterval` | Promo2가 진행되는 월 이름 (예: `"Jan,Apr,Jul,Oct"`) |
    
    | 컬럼명 | 설명 |
    | --- | --- |
    | `Id` | 샘플 고유 ID (예측 결과 제출용) |
    | `Store` | 매장 ID |
    | `DayOfWeek` | 요일 |
    | `Date` | 날짜 |
    | `Open` | 매장 개점 여부 (결측 가능) |
    | `Promo` | 프로모션 여부 |
    | `StateHoliday` | 공휴일 여부 |
    | `SchoolHoliday` | 방학 여부 |

## 1-2. 데이터 전처리: 결측치 처리 및 Feature Selection

1. 결측치 처리
    - Date 칼럼으로 Month, Day 등의 날짜 파생 변수 생성
    - Store 기준으로 train.csv와 store.csv를 병합하여 파일 재구성
    - 참조 코드 : [PreProcess_Merge.ipynb](Documents/GitHub/03.-ML-Model-Time-Series-Sales-Prediction/전처리/PreProcess_Merge.ipynb)
        
   <img width="616" alt="image" src="https://github.com/user-attachments/assets/d9af1ce7-f619-4339-9a02-261df4241d93" />    
    

1. Feature Selection
    - 제외한 대표적 Feature
        
    <img width="585" alt="image" src="https://github.com/user-attachments/assets/a3c0cc55-c225-425c-aeec-ce1513756ea7" />

    - 각 모델에 맞는 features 선택 및 encoding 진행 후 모델 학습 진행
        <img width="584" alt="image" src="https://github.com/user-attachments/assets/935e0175-1392-4e65-82ed-663c6a8af381" />

       
        
        ```python
        features = [
            'DayOfWeek',           # 요일 (매출 패턴 주기에 중요)
            'Month', 'Day',        # 날짜 정보 (계절성 반영)
            'Promo', 'Promo2',     # 프로모션 여부 (매출 상승 영향)
            'SchoolHoliday',       # 학교 방학 (특정 지역 매출에 영향)
            'StoreType',           # 매장 유형 (유통 방식에 따른 차이)
            'Assortment',          # 상품 구색 (기본/확장 여부)
            'CompetitionDistance'  # 경쟁 매장 거리 (매출에 반비례 영향)
        ]
        ```
        
    

<aside>

## 1-3. EDA 를 통한 가설설정 및 문제정의

**💡문제 정의💡**

이 보고서에서는 상점별 매출액 예측 문제를 다룹니다. 매출액 예측은 매출 증가를 위한 전략 수립, 마케팅 효과 분석, 인력 및 재고 관리 최적화 등을 위해 중요한 분석 과제입니다. EDA(탐색적 데이터 분석) 결과를 바탕으로, 다양한 변수들이 매출액에 미치는 영향을 분석하고, 이를 통해 매출액 예측 모델을 구축하는 것이 목적입니다.

**💡EDA 결과💡**

**✅ 기초 통계량 (Feature별 기초 통계)**

- 기초 통계량을 통해 변수들의 분포와 이상치를 확인
- **매출액**은 큰 범위의 값들을 가짐을 알 수 있으며, **CompetitionDistance**와 같은 변수들은 일부 상점에서 매우 높은 값을 가짐
      ![describe_output](https://github.com/user-attachments/assets/0d196687-0591-4169-b3fc-fd890d36c72a)


 ✅ **매출액과 타 변수 간의 관계 분석**

- **히트맵 분석**:
    - **Promo**와 **DayOfWeek** 변수는 매출액과 높은 상관관계를 보였습니다. 이 두 변수는 매출 예측에서 중요한 역할을 할 것으로 예상
    - 
        ![상관관계heatmap](https://github.com/user-attachments/assets/82a76cb7-f162-450e-b6d1-4c534f9fac21)

        
- **PairPlot 분석**
    - 매출액은 **상점 유형(StoreType)**에 따라 큰 차이를 보였습니다. 특히 **Promo** 변수는 매출에 중요한 영향을 미치는 변수로, 프로모션이 발생한 기간의 매출이 높게 나타남
    - **경쟁사 거리(CompetitionDistance)**는 상점 유형에 따라 차이가 있음을 알 수 있으며, 상점의 위치가 매출에 영향을 미친다는 것을 시사
        ![pairplot (1)](https://github.com/user-attachments/assets/8bbcb498-392f-483e-930d-1d2bf226900e)


**✅ ANOVA 분석 결과**
    <img width="1015" alt="ANOVA" src="https://github.com/user-attachments/assets/820772b6-7f9e-40de-bae4-ffedc4c749e3" />



- ANOVA 분석은 범주형 변수들이 매출액에 미치는 영향을 평가하는데 사용.
- **F-statistic** 값이 높을수록 그룹 간 차이가 크며, 이 값과 **p-value**를 함께 고려하여 변수들이 매출에 유의미한 영향을 미친다는 결론을 내릴 수 있습니다.
    - **주요 변수들**: `Promo`, `StoreType`, `Assortment`, `StateHoliday`, `DayofWeek`, `Month`, `Weekend vs Weekday` 변수들은 매출에 유의미한 영향
    - **SchoolHoliday**는 매출액에 큰 영향을 미치지 않는 변수
    
    <img width="871" alt="Boxplot" src="https://github.com/user-attachments/assets/b53b8a96-a791-49c3-84d2-66ba2957a692" />

    

**✅ 매출액 분포 확인**

- **휴일에 따른 프로모션과 매출액 분포** (Holiday Type | 0= 휴일 아님, 1 = 휴일)
    - **휴일 :** 스토어 타입에 따라 휴일 여부에 따른 프로모션 진행여부의 성과가 다르며, 특히 스토어 타입 b의 경우 휴일&프로모션 기간에 프로모션의 매출 기여가 큼

      <img width="869" alt="Holiday Promotion Sales" src="https://github.com/user-attachments/assets/0fa6802b-8d41-479b-9cf2-b4229939aada" />

    
    
    
    - **요일별: 월~금**에는 프로모션이 매출을 증가시키는 효과가 있지만, **주말**에는 프로모션 매출이 상대적으로 적음. StateHoliday의 매출은 SchoolHoliday의 매출보다 크고, SchoolHoliday에는 프로모션이 사용된 매출의 비중이 더 큼.
        
      <img width="1194" alt="DoD" src="https://github.com/user-attachments/assets/b08d77fe-6c7d-4524-89b7-700d71232160" />

        
- **스토어 타입에 따른 매출액 분포:**
    - **스토어 타입 B**가 전반적으로 매출이 다른 스토어 타입보다 높으며, 가장 넓은 매출 분포를 보임
    
    <img width="867" alt="Store Type" src="https://github.com/user-attachments/assets/dcabc12c-cde7-4055-b761-7e60479418b3" />


    

### **매출 예측을 위한 모델링 필요성**

EDA 결과를 바탕으로 매출 예측 모델을 구축하는 것이 필요합니다. 주요 분석 결과에서 확인된 **Promo**, **StoreType**, **Assortment**, **StateHoliday**, **DayofWeek**, **Month** 등의 변수는 매출액에 유의미한 영향을 미칩니다. EDA의 결과를 바탕으로 정의한 문제 해결을 위해 **회귀 분석** 또는 **머신러닝 모델**을 사용하여 **매출 예측 모델**을 구축하고, 각 변수의 중요도를 평가를 목표로 합니다. 

**가설 설정**

1. 요일(DayOfWeek), 프로모션(Promo), 직전 매출(Sales_lag)과 같은 시간 및 이벤트 변수는 매출에 유의미한 영향을 줄 것이다.
2. 트리 기반의 머신러닝 모델(Random Forest, XGBoost)은 선형 회귀보다 더 우수한 예측 성능을 보일 것이다.
3. 과거 매출 정보를 포함한 lag feature와 매출액 이동평균은 예측 정확도를 높이는 데 기여할 것이다.

**예측 모델 개발 방향**

- **모델 선택**: **선형 회귀 모델**을 기본으로 시작하고, 필요에 따라 **랜덤 포레스트** 또는 **XGBoost**와 같은 고급 모델을 사용
- **모델 평가**: 모델의 성능을 평가하기 위해 **MAE**, **RMSE**, **R²** 등의 지표를 활용
- **예상 활용도**: 매출 예측을 위한 주요 변수를 확인 후 매출 예측 모델을 구축하여 실제 매출을 예측하고, 이를 통해 매출을 최적화하기 위한 전략을 수립할 수 있습니다.
</aside>

## 2-1. **예측 모델 구현을 위한 학습 진행 및 결과 확인**

## Linear Regression - Ridge, Lasso

```python
features = ['DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Promo2',
            'CompetitionDistance',
            'Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14',
            'Sales_roll_mean_7','Day', 'Month', 'Year'
           ]
```

**[모델 생성 > 훈련 > 예측 과정]**  

<aside>

Step 1. 데이터 전처리: 시계열 특성 정리 

- 시계열 특성 계수 추가
    
    하루 전, 일주일 전, 2주 전 , 이동평균 시계열 특성 계수 추가
    
    **[시계열 특성 결과 해석 ]**
    
    1. **계절성 강도 측정**: 7일, 14일 전 데이터의 높은 계수는 강한 주간 계절성.
    2. **단기 vs 장기 영향력**: 시간이 지날수록(1일→7일→14일) 계수 패턴이 변화하는 양상을 확인할 수 있음
    3. **외부 요인과의 상호작용**: 휴일이나 프로모션과 같은 요인들이 시계열 패턴에 어떤 영향을 미치는지 볼 수 았음
    
- 결측치 제거
- 범주형 데이터: 원핫 인코딩 사용
- 정규화할 연속형 변수 선정 후 정규화 적용
- MinMaxScaler방법으로 스케일링

Step 2. 훈련 세트와 테스트 세트로 나누어서 학습 

- 다양한 회귀 모델 시도
- LinearRegression, Ridge 회귀 (다양한 alpha 값), Lasso 회귀 (다양한 alpha 값)
- 부트스트랩 샘플링을 사용하여 각 반복마다 훈련 데이터의 변형된 버전 사용
- 각 반복마다 모델 유형, MSE, RMSE, R2 점수 출력하여 최고 성능의 모델 채택
    
    ```python
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import random
    
    # 데이터 로드
    train_store = pd.read_csv("/content/train_store.csv")
    test = pd.read_csv("/content/test.csv")
    
    # 날짜 정렬 (Store별로 시계열 순서 유지)
    train_store.sort_values(['Store', 'Date'], inplace=True)
    
    # 시계열 특성 생성
    train_store['Sales_lag_1'] = train_store.groupby('Store')['Sales'].shift(1)
    train_store['Sales_lag_7'] = train_store.groupby('Store')['Sales'].shift(7)
    train_store['Sales_lag_14'] = train_store.groupby('Store')['Sales'].shift(14)
    train_store['Sales_roll_mean_7'] = train_store.groupby('Store')['Sales'].shift(1).rolling(window=7).mean().reset_index(0, drop=True)
    
    # 결측치 제거
    train_store.dropna(subset=['Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14',
                               'Sales_roll_mean_7'], inplace=True)
    train_store['StateHoliday'] = train_store['StateHoliday'].replace({'0': 0, 'a': 1, 'b': 1, 'c': 1}).astype(int)
    
    # 원핫인코딩 적용(범주형)
    categorical_cols = ['StoreType', 'Assortment']
    train_store_encoded = pd.get_dummies(train_store, columns=categorical_cols, drop_first=True)
    
    # 정규화할 연속형 변수 정의
    continuous_features = ['CompetitionDistance', 'Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14',
                           'Sales_roll_mean_7']
    
    # 정규화 적용
    scaler = MinMaxScaler()
    train_store_encoded[continuous_features] = scaler.fit_transform(train_store_encoded[continuous_features])
    
    # 특성과 타겟 변수 정의
    
    features = ['DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Promo2',
                'CompetitionDistance',
                'Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14',
                'Sales_roll_mean_7','Day', 'Month', 'Year'
               ]
    
    # 원핫인코딩된 열 이름 추가
    features.extend([col for col in train_store_encoded.columns
                    if col.startswith(tuple(f'{c}_' for c in categorical_cols))])
    
    # 특성과 타겟 분리
    X = train_store_encoded[features]
    y = train_store_encoded['Sales']
    
    # RMSPE 함수 정의 (Root Mean Squared Percentage Error)
    def rmspe(y_true, y_pred):
        return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    
    # 전체 훈련 및 검증 세트 분리
    X_train_full, X_val, y_train_full, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 반복 학습 (10번 반복)
    n_iterations = 10
    best_mse = float('inf')
    best_model = None
    models = []
    performances = []
    
    for i in range(n_iterations):
        # 다양한 모델 시도
        if i % 3 == 0:
            # 기본 선형 회귀
            model = LinearRegression()
            model_name = "LinearRegression"
        elif i % 3 == 1:
            # Ridge 회귀 (알파값 랜덤)
            alpha = random.uniform(0.1, 10.0)
            model = Ridge(alpha=alpha)
            model_name = f"Ridge(alpha={alpha:.2f})"
        else:
            # Lasso 회귀 (알파값 랜덤)
            alpha = random.uniform(0.001, 1.0)
            model = Lasso(alpha=alpha)
            model_name = f"Lasso(alpha={alpha:.4f})"
    
        # 부트스트랩 샘플링 (데이터 변형을 위해)
        sample_indices = np.random.choice(len(X_train_full), size=int(0.8*len(X_train_full)), replace=True)
        X_train = X_train_full.iloc[sample_indices]
        y_train = y_train_full.iloc[sample_indices]
    
        # 모델 학습
        model.fit(X_train, y_train)
    
        # 모델 평가
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
    
        # 모델 및 성능 저장
        models.append((model, model_name))
        performances.append((mse, rmse, r2))
    
        # 최고 모델 업데이트
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_model_name = model_name
    
        print(f"Iteration {i+1}: {model_name} - MSE = {mse:.4f}, RMSE = {rmse:.2f}, R2 = {r2:.4f}")
    
    # 최고 모델로 훈련 및 검증 세트 예측
    print(f"\nBest Model: {best_model_name} (MSE = {best_mse:.4f})")
    
    # 훈련 세트에 대한 성능 평가
    y_train_pred = best_model.predict(X_train_full)
    train_mae = mean_absolute_error(y_train_full, y_train_pred)
    train_mse = mean_squared_error(y_train_full, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train_full, y_train_pred)
    train_rmspe = rmspe(y_train_full, y_train_pred)
    
    # 검증 세트에 대한 성능 평가
    y_val_pred = best_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val, y_val_pred)
    val_rmspe = rmspe(y_val, y_val_pred)
    
    # 성능 평가 결과 표시
    print(f"\n{'Metric':<10} {'Train':>10} {'Test':>10}")
    print(f"{'-'*30}")
    print(f"{'MAE':<10} {train_mae:>10.2f} {val_mae:>10.2f}")
    print(f"{'RMSE':<10} {train_rmse:>10.2f} {val_rmse:>10.2f}")
    print(f"{'R2 score':<10} {train_r2:>10.2f} {val_r2:>10.2f}")
    print(f"{'RMSPE':<10} {train_rmspe:>10.2f} {val_rmspe:>10.2f}")
    ```
    

Step 3. 모델 성능평가 지표 확인

- 최종적으로 성능이 가장 좋은 선형회귀 모델을 선택
- 훈련 및 검증 세트에 대한 상세한 성능 지표 출력

    ![LR score](https://github.com/user-attachments/assets/ae35245c-7874-4990-ac37-637d47b21663)


</aside>

**[성능 평가]**

- 성능평가지표 설명
    - MAE: 예측값과 실제값 사이의 평균 절대 오차
    - MSE (Mean Squared Error): 예측값과 실제값의 차이를 제곱하여 평균한 값. 이 값이 크다는 것은 예측 오차가 크다는 의미
    - RMSE: MSE의 제곱근을 취한 값으로, 예측 오차의 크기를 실제 단위. 이 값이 작을수록 모델의 예측이 더 정확.
    - MAPE:  예측값과 실제값의 차이를 실제값에 대해 백분율로 나타내는 지표
    : MAPE 값이 크다는 것은 모델이 예측하는 과정에서 매우 큰 오차가 발생했다는 것을 의미. 특히, MAPE 값이 이렇게 비정상적으로 크다면 데이터에 0에 가까운 값이나 극단적인 값이 포함되어 있을 가능성이 있음. 이런 극단값이 MAPE에 크게 영향을 미침
    - R²:  모델이 실제값을 얼마나 잘 설명하는지 나타내는 지표입니다. 0과 1 사이의 값을 가지며, 1에 가까울수록 모델이 데이터를 잘 설명한다고 볼 수 있습니다. 0.6326은 모델이 약 63%의 변동성을 설명한다는 의미입니다. 이는 모델이 어느 정도 유효하다는 것을 나타내지만, 개선할 여지가 있음

        | **Metric** | Train | Test |
        | --- | --- | --- |
        | **MAE** | 1952.87 | 1950.04 |
        | **RMSE** | 2585.43 | 2579.74 |
        | **R2 score** | 0.55 | 0.55 |
        | **RMSPE** | 0.38 | 0.42 |

**[설명]** 

- 다양한 회귀분석 모델을 시도 하였으나, R2 점수가 0.55로 비교적 낮은 것으로 보아, 모델의 전반적인 예측 성능을 향상시킬 여지가 있음.
    - Decision Tree, Random Forest, XGBoost 등 고급 트리모델로 예측 모델을 설계할 필요성이 있음을 발견.
    - 현재 상태에서도 활용 가능한 모델이지만, 특성 엔지니어링이나 다른 알고리즘을 추가적으로 시도하여 성능을 개선할 필요가 있음.
- **판매량(Sales) 데이터 변동성의 약 55%를 설명,** 모델이 설명하지 못하는 변동성이 45%.
    - MAE는 (예측값과 실제값 사이의 평균 절대 오차)  모델의 예측값이 실제값과 평균적으로 약 1950.04(달러) 만큼 차이가 나는 것을 확인.
- **과적합 여부 판정:** Train과 Test 데이터에서 MAE, RMSE, R2 점수가 매우 유사한 값을 보이기 때문에 모델이 훈련 데이터에 과적합되지 않았다고 판단.

**[실제 매출과 예측 매출 간 산점도, 시계열 예측, 잔차 분포 분석]**

- **산점도**
    - 대체로 잔차 산점도의 분포가 예측 기준선에 밀집해 있어 어느정도 매출 예측 모델이 성능이 있음을 확인할 수 있음.
    - 매출이 커질수록 정확도가 떨어지고, 매출이 0 에 가까울수록 예측 정확도가 떨어짐.
    
    ![LR 산점도](https://github.com/user-attachments/assets/ad05c996-f983-4960-b7c4-41f5246e1c35)

    
- **잔차 분석: 잔차의 분포와 정규성 검정**
    - 하지만, 잔차 분포 히스토그램의 분포가 타 ML 예측 모델과 비교하여 넓게 분포되어 있음. Q_Q Plot 으로 잔차의 정규성을 확인 해본 결과, 잔차의 정규성 문제로 인해 기존의 선형 회귀 모델보다 더 적합한 예측 모델 생성이 필요함.
    
    ![qqplot](https://github.com/user-attachments/assets/41c7a8ec-89cc-4134-b61b-a92db38fbf45)

    ![Residual](https://github.com/user-attachments/assets/81b3bfd6-07bb-4c59-bbc0-6bdc5f68a0dc)

    

- **Actual vs Predicted (Date 순서)**
    - 정확도는 타 모델에 비해 낮지만, 매출 예측의 트랜드는 비슷하게 예측
    
    ![Actual Predicted](https://github.com/user-attachments/assets/a2442103-1314-4fec-a838-dc8edf5045d2)


**[중요 독립변수 해석]**

- **각 특성(feature)의 계수(coefficient)와 절대값(abs_coefficient)**
    
    <img width="379" alt="feature coeff" src="https://github.com/user-attachments/assets/3bb48489-9a4b-4a7b-9cac-c2e1ece003f6" />

    <img width="655" alt="top10features" src="https://github.com/user-attachments/assets/edbede44-2d49-4859-be3b-ae024329000d" />


    <img width="627" alt="image" src="https://github.com/user-attachments/assets/81e96698-40f7-44bc-85dc-b502310ddd62" />


## Decision Tree

**[모델 생성 > 훈련 > 예측 과정]** 

- 모델 생성: 결정 트리 모델
- 훈련: store_train 데이터를 사용해서 모델을 학습하였음
- 예측: 테스트 데이터 입력 → 학습된 모델이 결과 예측
- 성능 평가(MAE, RMSE, R2, RMSPE)

```python
# X, y 추출
# X엔 우리가 정한 feature 리스트만 넣음
features = [
    'Store',
    'DayOfWeek',           # 요일 
    'Month', 'Day',        # 날짜 정보
    'Promo', 'Promo2',     # 프로모션 여부 
    'SchoolHoliday',       # 학교 방학 
    'StoreType_b', 'StoreType_c', 'StoreType_d',    # 원-핫 인코딩된 StoreType 변수들
    'Assortment_b', 'Assortment_c',  # 원-핫 인코딩된 Assortment 변수들
    'CompetitionDistance'  # 경쟁 매장 거리
]
```

<aside>

데이터 전처리: 

```python
# 원-핫 인코딩을 'StoreType'과 'Assortment' 열에 적용
df = pd.get_dummies(df, columns=['StoreType', 'Assortment'], drop_first=True)
```

```python
# feature 리스트에 없는 변수들은 삭제
df.drop(["Promo2SinceYear", "Promo2SinceWeek", "PromoInterval"], axis=1, inplace=True)
```

</aside>
        
**[성능 평가]**
        
        | **Metric** | Train | Test |
        | --- | --- | --- |
        | **MAE** | 1624.70 | 1,657.48 |
        | **RMSE** | 2447.11 | 2,490.08 |
        | **R2 score** | 0.5951  | 0.5899 |
        | **RMSPE** | 0.42 | 0.39 |

**[설명]**

- **과적합 여부 판정**
    - Train과 Test 데이터에서 MAE, RMSE, R2 점수가 매우 유사한 값을 보이기 때문에 모델이 훈련 데이터에 과적합되지 않았다고 판단.
- **R² score**
    - 훈련 데이터와 테스트 데이터 모두에서 약 0.59로, 모델이 전체 변동성의 약 59%를 설명하고 있음을 의미. 모델의 예측 성능은 어느 정도 있지만, 개선의 여지가 존재.
- **RMSE (Root Mean Squared Error)**
    - 훈련 데이터와 테스트 데이터에서 RMSE 값은 각각 2,447.11과 2,490.08로 비슷하며, 이 역시 예측이 잘 이루어졌음. 오차 크기는 평균적으로 2,450달러 내외로 측정
- **RMSPE (Root Mean Squared Percentage Error)**: 훈련 데이터에서 0.42, 테스트 데이터에서 0.39로, 상대적인 오차가 약 40% 내외로 나타남. 이는 예측값의 상대적인 정확도가 일정 수준 유지되고 있음을 보여줌

**[실제 매출과 예측 매출 간 산점도, 시계열 예측, 잔차 분포 분석]**

- **산점도**
    - 매출이 커질수록 정확도가 떨어지고, 매출이 0 에 가까울수록 예측 정확도가 떨어짐
    
    ![actual_predict](https://github.com/user-attachments/assets/d629ebfb-d5e5-4971-aaf1-a8e07e3d9734)

    
- **Actual vs Predicted (Date 순서)**
    - 예측값이 실제값을 잘 따라가지만, 진폭이 더 큽니다. 추세는 잘 맞추지만, 변동성을 과장하고 있습니다. 예측값의 변동성을 줄이는 모델 튜닝이 필요
    
    
    ![date](https://github.com/user-attachments/assets/6e0d7e30-46ca-4677-a22b-0d61a34ce497)


**[중요 독립변수 해석]**  
    ![중요변수](https://github.com/user-attachments/assets/62c3344c-b054-431f-9b40-686e2dfacc1a)




| 순위 | 변수 | 해석 |
| --- | --- | --- |
| 1 | **DayOfWeek** | 0.6의 중요도로 가장 높은 영향력. 주말과 평일의 매출 차이가 클 가능성이 높음. |
| 2 | **Promo** | 0.15의 중요도로 두 번째로 중요한 특성입니다. 프로모션 진행 여부가 매출에 상당한 영향을 미침. |
| 3 | **CompetitionDistance** | 약 0.07의 중요도로, 매장과 경쟁 업체 간의 거리가 매출에 영향을 준다는 것을 보여줌 |
| 4 | **StoreType_b** | 약 0.05의 중요도를 보이며, 특정 매장 유형이 매출 예측에 기여 |
| 5 | **Store** | 0.03-0.04의 중요도를 보이며, 개별 매장의 특성과 계절적 요인도 매출에 영향 |
| 6 | **Month** | 이하상동 |
| 7 | **Promo2** | 연중반복되는 프로모션은 상대적으로 낮은 중요도를 가짐.  |
| 8 | **Day** | 상대적으로 낮은 중요도 |
| 9 | **Assortment_c** | 상대적으로 낮은 중요도 |
| 10 | **SchoolHoliday** | 상대적으로 낮은 중요도 |

**[결론]**

- 결정 트리 모델로 매출 예측을 수행한 결과, 모델은 매출 변동의 약 59%를 설명하며 훈련 및 테스트 데이터에서 일관된 성능을 보임. 특히 요일이 가장 중요한 예측 변수(중요도 0.6)로 확인되었으며, 프로모션과 경쟁업체와의 거리도 중요한 요소.
- **MSE**, **RMSE**, **MAE** 등은 예측값과 실제값 간의 차이를 평가하는 데 유용하지만, 매출과 같은 **숫자 규모가 큰 데이터**에서 예측 오차가 상대적으로 크게 나타날 수 있음. MAE, RMSE와 같은 절대적 지표보다는 대규모 매출 데이터에서 RMSPE와 같은 상대적 지표가 더 적합했으며, 단일 트리 모델의 한계가 있음. 향후 하이퍼파라미터 튜닝이나 앙상블 모델 적용을 통해 예측 성능을 개선할 수 있을 것으로 기대됨.

## Random Forest

```python
# 사용 features
features = [
    'DayOfWeek',           # 요일 (매출 패턴 주기에 중요)
    'Month', 'Day',        # 날짜 정보 (계절성 반영)
    'Promo', 'Promo2',     # 프로모션 여부 (매출 상승 영향)
    'SchoolHoliday',       # 학교 휴일 (특정 지역 매출에 영향)
    'StoreType',           # 매장 유형 (유통 방식에 따른 차이)
    'Assortment',          # 상품 구색 (기본/확장 여부)
    'CompetitionDistance',  # 경쟁 매장 거리 (매출에 반비례 영향)
    'StateHoliday',          # 주 휴일 (학교 휴일과 같은 이유로 추가)
    'Sales_lag_1', 'Sales_lag_7', 'Sales_roll_mean_7' # 1일, 7일, 7일 평균 판매액 추가
]
```

**[모델 생성 > 훈련 > 예측 과정]**  

<aside>

Step 1. 데이터 전처리

- 수치형, 범주형 변수 구분
- 범주형의 경우 One Hot Encoding으로 인코딩 진행
- 더 정확한 예측 위해 1일전, 7일전 판매액 등 Sales_lag feature 추가

Step 2. train, test 데이터셋 분리 후 학습 진행

Step 3. 학습 후 성가지표 확인

- Random Forest 파라미터 설정
    
    ```python
    from sklearn.ensemble import RandomForestRegressor
    
    rf = RandomForestRegressor(
        n_estimators= 100,
        random_state= 42
    )
    rf.fit(X_train, Y_train)
    ```
    
- MAE, RMSE, R2, RMSPE 점수 출력
- 시각화로 해석

Step 4. Cross Validation으로 추가 검증 진행

</aside>

**[성능 평가]**
        
        | Metric | Test | Train |
        | --- | --- | --- |
        | MAE | 826.97 | 372.02 |
        | RMSE | 1507.09 | 788.73 |
        | R2 score | 0.847 | 0.95 |
        | RMSPE | 0.36 | 0.14 |

**[설명**]

- **Hold Out 방식으로 진행한 결과 Sales 평균 대비 26%의 변동성을 가짐** 또한 약 **85% 예측 정확도**를 보임
- **과적합 여부 판정 (Test vs Train)**
    - 일반적으로 2배정도의 RMSE 차이는 괜찮으나, **약간의 과적합이 존재**한다고 해석 가능
- Cross Validation (참조)
    - K = 5로 설정 후 진행
        
        ```python
        from sklearn.model_selection import cross_val_score
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        scores = cross_val_score(rf, encoded_X, Y, cv=5, scoring='neg_mean_squared_error')
        
        # RMSE로 변환
        rmse_scores = np.sqrt(-scores)
        
        print("각 Fold의 RMSE:", rmse_scores)
        print("평균 RMSE:", rmse_scores.mean())
        print("표준편차:", rmse_scores.std())
        ```
        
    
    | 각 Fold의 RMSE | [1328.04, 1432.47,  1294.74, 1494.75, 1417.60] |
    | --- | --- |
    | 평균 RMSE | 1393.52 |
    | 표준편차 | 76.65 |
    | RMSE 대비 Sales 평균 | 0.24 |
    - **예측값이 평균대비 24% 변동성을 가짐**
        - Hold Out과 변동성이 비슷하기에 특정 데이터 분할에 크게 의존하지 않음을 의미
        - **안정적이고 일반화 성능이 우수한 모델**

**[실제 매출과 예측 매출 간 산점도, 시계열 예측, 잔차 분포 분석]**

- **산점도**
    ![Sales vs Predict Sales 산점도 (1)](https://github.com/user-attachments/assets/55a9c677-b911-4efb-b88c-96115e0c9147)
    
    - 대체로 기준선에 밀집해 있어 예측이 잘 수행됨
    - 그러나 매출이 커질수록 예측 정확도가 떨어지고, Actual Sales가 0일 때 예측 정확도 하락
    
- **잔차 분석**
    
    ![image (37)](https://github.com/user-attachments/assets/7fb83ab9-b041-4667-8134-3a9572dd8506)

    
    - 대부분 오차가 0에 몰려있고, **정규분포에 가까운 형태**를 보이고 있어 편향이 심하지 않고, 좋은 모델이라는 것이 보임
    - 꽤 큰 이상치 값이 존재하지만 대부분 정규분포 형태
    
- **Actual vs Predicted (Date 순서)**
    
    ![image (38)](https://github.com/user-attachments/assets/abb5a229-1d73-4f54-af0f-25c5df1590ed)

    
    - 전반적 패턴 추종 우수
    - 모델이 큰 흐름을 잘 따라감. 대체적으로 같은 방향으로  움직이고 있어, 계절성 흐름을 잘 보여줌.

**[중요 독립변수 해석]**

    ![영향을 미치는 주요변수 (1)](https://github.com/user-attachments/assets/e6bafd99-d15f-4e70-929c-7b07c743d9ce)


<img width="576" alt="image" src="https://github.com/user-attachments/assets/7d684968-d65b-4590-9aa1-eca0da744541" />


## XGBoost

```python
features = [
    'Store', 'DayOfWeek', 'Month', 'Day',
    'Promo', 'Promo2', 'SchoolHoliday',
    'StoreType', 'Assortment',
    'CompetitionDistance',
    'Sales_lag_1', 'Sales_lag_7', 'Sales_roll_mean_7'
]
```

<aside>

Step 1. 데이터 전처리

- 매장의 과거 매출 패턴을 반영하기 위해 시계열 특성 생성

| 생성 변수 | 설명 |
| --- | --- |
| Sales_lag_1 | 하루 전 매출 |
| Sales_lag_7 | 7일 전 매출 |
| Sales_roll_mean_7 | 직전 7일간 평균 매출 |
- 범주형 인코딩
    - StoreType, Assortment은 LabelEncoder를 사용하여 수치형으로 변환
    
        ⇒ XGBoost는 범주형 변수를 직접 처리하지 않기 때문에 수치 변환이 필요
    
- 결측치 처리
    - 시차 변수 생성 과정에서 발생한 NaN 값은 제거하여 학습의 신뢰성 확보

Step 2. 모델 학습 구성

- 사용 모델: XGBoost Regressor (Booster)
- 사용 데이터: tain_store.csv
- 훈련-검증 분할: train_test_split(test_size = 0.2, random_state = 42)
- 주요 파라미터 설정:
    
    ```python
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'eta': 0.1,
        'seed': 42
    }
    ```
    

- max_depth 하이퍼 파라미터 튜닝
  
    ![image (37)](https://github.com/user-attachments/assets/99a2cdfe-90d3-4265-935a-ee1a3228e518)


- 그래프 해석
    - Validation RMSE는 max_depth = 6 ~ 7까지 꾸준히 감소하다가 이후 **완만한 감소 또는 정체**
    - max_depth = 9 ~ 10에서는 과적합 가능성 관찰: Train RMSE는 크게 낮지만 Validation RMSE는 큰 개선 없음
- **최적 max_depth = 6**
    - 학습/검증 성능의 균형이 가장 잘 맞는 구간
    - 모델 복잡도 대비 검증 성능이 안정적이며 일반화 가능성 우수

Step 3. 성능 평가 지표 확인

- MAE, RMSE, R2, RMSPE 점수 출력

![image (38)](https://github.com/user-attachments/assets/f995c43b-2665-4ff5-8dfb-bd757101da1e)

</aside>

**[성능 평가]**

        | **Metric** | Train | Test |
        | --- | --- | --- |
        | **MAE** | 642.22 | 643.70 |
        | **RMSE** | 1069.47 | 1073.60 |
        | **R2 score** | 0.92 | 0.92 |
        | **RMSPE** | 0.23 | 0.16 |

**[설명]**

- Train과 Test 간의 모든 지표의 차이가 매우 작음 → 과적합(Overfitting) 되지 않았음을 시사
- **높은 R²(0.92)** 와 **낮은 RMSE/MAE**는 이 모델이 **정확하면서도 일반화된 예측 성능**을 가짐
- RMSPE 관점에서 검증 데이터에서 RMSPE가 더 낮은 것은 모델이 다양한 매출 규모에서도 안정적인 비율로 오차를 유지

**[실제 매출과 예측 매출 간 산점도, 시계열 예측, 잔차 분포 분석]**

- **산점도**
    - 대부분의 데이터가 대각선 근처에 분포 → 예측이 전반적으로 잘 맞음
    - 저매출 구간(0 ~ 5000)에서 데이터가 밀집되어 있고 다소 산개 → 저매출일수록 예측 정확도가 낮아지는 경향
    - 중매출 구간(5000 ~ 25000)에서 대부분 예측이 잘 일치하며 모델의 성능이 가장 안정적인 구간
    - 고매출 구간(25000 이상)에서 일부 예측값이 실제보다 낮음 → 고매출일수록 과소 예측 경향 존재
    ![image (39)](https://github.com/user-attachments/assets/985e23a0-59fb-4362-bc81-18d9ac34455a)
  

- **Actual vs Predicted (Date 순서)**
    - 전반적인 패턴 추종 우수:
        
        예측값(Predicted)은 실제값(Actual)과 매우 유사한 추이를 보이며 대부분의 상승·하강 구간에서 흐름을 유사하게 반영
        
    - 피크 구간 예측력 양호:
        
        매출 급감/급증 시점에서도 모델이 급격한 추세 변화를 잘 학습한 것으로 보임
        
    - 예측 과대/과소 구간 미미:
        
        일부 구간에서 예측치가 실제치보다 소폭 낮거나 높은 경향이 있으나, 전반적인 예측 오차는 제한적
        
    ![스크린샷 2025-03-31 111535](https://github.com/user-attachments/assets/2b67413a-785d-4fd0-b978-f1eaad7a1a67)


- **잔차 분석**
    - 잔차가 정규 분포 형태에 가까워 잔차(실제-예측)의 중심이 0 근처에 대칭적으로 몰려있다
    - 모델이 예측을 전반적으로 잘하고 있음
    
    ![image (40)](https://github.com/user-attachments/assets/0ba5aa73-fa42-4f59-a1b6-7a75528fffdb)

    

**[중요 독립변수 해석]**

    ![image (37) 복사본](https://github.com/user-attachments/assets/610138ad-6680-4835-b6cc-99d674b2a96f)


| **순위** | **변수** | **해석** |
| --- | --- | --- |
| 1 | **Sales_lag_7** | 일주일 전 매출: 요일 주기를 강하게 반영함(가장 중요) |
| 2 | **Day** | 일자 정보: 월 초/말, 특정 날짜(예: 15일, 30일 등)의 영향 반영 가능 |
| 3 | **Store** | 매장 ID: 매장별 특성 반영. 고유 매출 패턴이 있음 |
| 4 | **Sales_lag_1** | 어제 매출: 전일 영향을 많이 받음. 매출의 연속성 반영 |
| 5 | **Sales_roll_mean_7** | 최근 7일 평균 매출: 매출 추세를 반영함 |
| 6 | **CompetitionDistance** | 경쟁 매장 거리: 가까울수록 매출에 부정적 영향 가능 |
| 7 | **Month** | 월 정보: 계절성 또는 월별 패턴 반영 |
| 8 | **DayOfWeek** | 요일 정보: 요일별 매출 차이 반영(예: 주말, 평일) |

## 3. 결론: 모델 성능 비교 및 해석 & 최종 모델 선정

### 3-1.  구현된 매출 예측 모델 결과

**[각 모델 성능 비교]**

| 평가기준 | Linear Regression | Decision Tree | Random Forest | **XGBoost** |
| --- | --- | --- | --- | --- |
| MAE | 1950.04 | 1624.70 | 826.97 | **643.7** |
| RMSE | 2579.74 | 2447.11 | 1507.09 | **1073.06** |
| R²  | 0.55 | 0.60 | 0.85 | **0.92** |
| RMSPE | 0.42 |  0.39 | 0.32 | **0.23** |

    - 참고 - 평가기준 설명
    - MAE: 실제값과 예측값의 절대 오차의 평균(값이 작을수록 좋음)
    - RMSE: 오차의 제곱 평균에 루트를 씌운 값(값이 작을 수록 좋음)
    - R²: 결정계수, 1에 가까울수록 모델이 데이터를 잘 설명한다는 의미(값이 높을수록 좋음)
        
        **⇒ XGBoost 모델을 최적 모델로 설정**
        
    - RMSPE: 예측 오차를 실제값 대비 비율로 계산해서 평균한 뒤, 제곱근을 취한 지표 (값이 작을 수록 좋음)

            
            
            

### 3-2.  최종 모델 선정 : XGBoost 모델

**[모델 요약: Feature Importance]**

![image (41)](https://github.com/user-attachments/assets/9769fcca-ff3d-4a03-8154-a45a8583c491)


- 매출 연속성: Sales_lag_7과 Sales_lag_1은 매출 예측에 중요한 역할을 하며, 과거 매출 데이터를 잘 반영
- 경쟁 환경: Store와 CompetitionDistance는 매장 특성 및 경쟁 상황을 고려한 중요한 요소.
- 날짜 및 요일: Day와 DayOfWeek는 날짜별, 요일별 매출 패턴을 반영.
- 추세 및 계절성: Sales_roll_mean_7과 Month는 매출 추세와 계절성을 반영.

**[SHAP 분석 인사이트]**

- 참고 - SHAP(Shapley Additive exPlanations)설명
    
    을 활용하면 각 특성이 예측에 미치는 영향을 정량적으로 파악
    
    1. **특성 중요도 평가:**
        - 각 특성이 예측에 얼마나 기여하는지 평가하여, 모델이 어떤 특성에 주로 의존하는지 파악
    2. **특성 영향력 분석:**
        - 특성의 값이 예측 결과에 미치는 영향을 시각화하여, 특정 특성이 예측을 어떻게 변화시키는지 이해
    3. **상호작용 효과 탐지:**

![image (37) 복사본 2](https://github.com/user-attachments/assets/2e334491-b2af-4c2d-9053-57b4028b62ed)



- 프로모션 (Promo): 프로모션은 매출을 높이는 중요한 요소로, 마케팅 효과를 뒷받침.
- 주말 매출: 주말은 매출이 낮아지는 경향, 주말 매출을 개선할 전략 필요.
- 연말 특수: 연말에 매출이 증가하는 경향, 계절성에 맞춘 전략 필요.
- 경쟁과 상권: 경쟁이 가까운 매장이 오히려 매출이 높은 경향, 상권 분석 및 경쟁 전략 강화 필요.

### 3-3. XGBoost 기반 향후 6주간 매출 예측

![image (38) 복사본](https://github.com/user-attachments/assets/98604140-d114-4c35-a125-c367a9388a54)


- 향후 6주간의 매출은 비선형적인 흐름을 보이며 2015년 8월 초에 매출이 급등할 가능성이 높음
- 전체적으로 예측 매출은 상승-하락-회복을 반복하는 변동성이 높은 흐름
- 매출 예측값: 예측값에 기반하여 **탄력적** 비즈니스 운영 전략 수립 시 활용 가능 ****

| 포인트 | 설명 |
| --- | --- |
| **예측된 매출 피크 주간에 프로모션 집중** | 8월 초 매출 피크가 예측되므로 해당 시기에 제품 **재고 확보, 마케팅 예산 증액** 등의 전략 고려 |
| **안정적 매출 구간은 비용 최적화 전략 적용 가능** | 8월 중순~9월 초에는 고정비 절감, 효율 운영 중심의 전략을 구성해볼 수 있음 |
