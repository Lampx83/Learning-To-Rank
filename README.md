# Step 1: PhoBERT Text Classification with PyTorch
Huấn luyện một mô hình phân loại văn bản bằng việc sử dụng mô hình ngôn ngữ PhoBERT và đánh giá hiệu suất của mô hình trên một tập dữ liệu kiểm tra.
### Dữ liệu
```csv
text,label
"Nguyễn Văn Đức là một giáo sư ngành xây dựng và giảng dạy tại đại học Xây dựng Hà Nội. Trong suốt sự nghiệp của tôi, tôi đã tham gia vào nhiều dự án xây dựng lớn, từ các công trình cầu đường cho đến nhà máy. Giáo sư đã có 100 bài báo và 4 giải thưởng",1
"Cuốn sách này nói về lịch sử của Trung Quốc, và là một tài liệu rất quý giá cho những ai quan tâm đến lịch sử và văn hóa của Trung Quốc.",0
"Công ty tôi đã thực hiện thành công nhiều dự án lớn trong lĩnh vực xây dựng cầu đường và nhà máy. Chúng tôi luôn đặt chất lượng sản phẩm lên hàng đầu, và tôi tự hào vì đã đóng góp vào những công trình đó.",0
```

## Kết quả chạy
```
Actual:   [0, 1, 1, 0, 1, 1, 0]
Predict:  [0, 1, 1, 0, 1, 1, 1]
              precision    recall  f1-score   support

  non-expert       1.00      0.67      0.80         3
      expert       0.80      1.00      0.89         4

    accuracy                           0.86         7
   macro avg       0.90      0.83      0.84         7
weighted avg       0.89      0.86      0.85         7
```
# Step 2: Extract Features
Trích xuất thông tin từ các hồ sơ chuyên gia và lưu trữ thông tin trong một Pandas DataFrame.
### Dữ liệu
Văn bản chuyên gia đã phát hiện được ở bước trên
### Kết quả chạy
```commandline
 education_level  years_of_experience  papers  awards
0                 4                    0       100       4
1                 2                   10        40       0
2                 0                    0         0       0
3                 0                    0        12       0
4                 0                    8         0       0
5                 0                    0        25       2
6                 3                    0        10       0
7                 0                    0         0       3
8                 3                    0         5       0
9                 4                    0         8       4
10                0                    0         0       0
11                0                    0         0       0
12                0                    0         0       0
13                0                    8        15       0
14                0                    6        11       0
15                0                   10        20       1
16                0                    7        15       0
```
# Step 3: Learning to Rank
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

### Dữ liệu
Dữ liệu huấn luyện và test trong file expert_data.csv. Các feature gồm có:
- **education_level**: Trình độ học vấn (1: Kỹ sư, 2: thạc sỹ, 3: tiến sỹ, 4: giáo sư)
- **years_of_experience**: Số năm kinh nghiệm
- **papers**: Số bài báo đã viết
- **awards**: Số giải thưởng (có thể thay bằng số chứng chỉ)
File dữ liệu mẫu như sau:
```data
expert_id,education_level,years_of_experience,papers,awards,relevance_score
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
### Kết quả chạy
Đánh giá các mô hình Learning to rank bằng cách tính toán các chỉ số đánh giá như: 
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R2 Score.
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