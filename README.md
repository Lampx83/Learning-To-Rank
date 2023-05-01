# Learning to Rank
Các mô hình Learning to rank được được thử nghiệm bao gồm:
- Linear Regression
- KNeighbors Regressor
- Decision Tree Regressor
- Support Vector Regression
- AdaBoost Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Bayesian Ridge
- MLP Regressor
- XGB Regressor
- CatBoost Regressor
- TensorFlow Regressor
- LightGBM Regressor

## Dữ liệu
Dữ liệu huấn luyện và test trong file expert_data.csv. Các feature gồm có:
- **education_level**: Trình độ học vấn (1: PhD, 2: Master, 3: Bachelor)
- **years_of_experience**: Số năm kinh nghiệm
- **projects**: Số dự án đã tham gia
- **awards**: Số giải thưởng (có thể thay bằng số chứng chỉ)
File dữ liệu mẫu như sau:
```data
expert_id,education_level,years_of_experience,projects,awards,relevance_score
1,2,5,10,0,0.7
2,1,8,20,2,0.9
3,2,3,5,1,0.6
4,1,12,50,5,1.0
5,3,2,2,0,0.5
6,1,15,30,3,0.95
7,2,7,15,1,0.8
8,1,10,25,2,0.85
9,3,1,0,0,0.4
10,2,4,8,0,0.65

```
## Cách sử dụng
### Chuẩn bị dữ liệu
Các dữ liệu đầu vào nên được lưu dưới dạng file CSV với tên "expert_data.csv" và các cột theo đúng thứ tự: expert_id, education_level, years_of_experience, projects, awards, relevance_score.
Sử dụng đoạn mã data = pd.read_csv('expert_data.csv') để tải dữ liệu từ file CSV vào biến data.
### Chia dữ liệu thành tập huấn luyện và tập kiểm tra
Sử dụng đoạn mã 
```
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
```
để chia dữ liệu thành tập huấn luyện và tập kiểm tra.
### Chuẩn bị dữ liệu cho việc huấn luyện
Sử dụng đoạn mã 
```
train_features = train_data.drop(['expert_id', 'relevance_score'], axis=1) 
train_target = train_data['relevance_score'] 
```
để tạo ra các đặc trưng huấn luyện và nhãn huấn luyện.
### Đánh giá các mô hình học máy
đánh giá các mô hình Learning to rank bằng cách tính toán các chỉ số đánh giá như: 
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R2 Score.

## Kết quả chạy
```
                             Year        MSE       MAE    R2 Score
Linear Regression            1940  0.010100  0.100481   0.838405
K Neighbor sRegressor        1970  0.021245  0.139952   0.660072
Decision Tree Regressor      1980  0.010000  0.100000   0.840000
Support Vector Regression    1995  0.027362  0.164467   0.562211
Ada Boost Regressor          1995  0.010000  0.100000   0.840000
Random Forest Regressor      1995  0.012996  0.108250   0.792062
Gradient Boosting Regressor  1999  0.007569  0.085836   0.878901
Bayesian Ridge               2000  0.016681  0.128719   0.733103
MLP Regressor                2006  0.034835  0.137175   0.442645
XGB Regressor                2014  0.006312  0.075316   0.899004
Cat Boost Regressor          2017  0.008764  0.088468   0.859770
TensorFlow Regressor         2021  3.220802  1.432241 -50.532838
Light GBM                    2022  0.073789  0.250000  -0.180625
```