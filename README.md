# Learning to Rank
Sử dụng Gradient Boosting, XGBoost và Tensorflow để xếp hạng dữ liệu.

## Giới thiệu về Learning to Rank
Learning to Rank (LTR) là một lĩnh vực trong Học máy/Học sâu (Machine/Deep Learning) và Tìm kiếm thông tin (Information Retrieval) nghiên cứu các phương pháp để xếp hạng các tài liệu dựa trên một số yếu tố nhất định.

## Giới thiệu về các thuật toán xếp hạng

### Gradient Boosting
Gradient Boosting là một thuật toán tập hợp các mô hình dự đoán đơn giản, chẳng hạn như cây quyết định, để tạo ra một mô hình dự đoán mạnh mẽ hơn. Thuật toán này tập trung vào việc tối thiểu hóa sai số của mô hình trên tập huấn luyện bằng cách thêm các mô hình mới và cập nhật trọng số các mẫu.

### XGBoost
XGBoost là một thuật toán Học máy đang được sử dụng rộng rãi trong cả Học sâu và Học máy cổ điển. Nó cũng là một loại thuật toán GBDT (Gradient Boosting Decision Tree) được tối ưu hóa để đạt được hiệu quả cao và thời gian chạy nhanh.

### Tensorflow
Tensorflow là một thư viện Học sâu (Deep Learning) được sử dụng rộng rãi trong nghiên cứu và ứng dụng thực tế. Nó cho phép người dùng xây dựng các mô hình Học sâu như mạng nơ-ron và sử dụng chúng để giải quyết các bài toán phức tạp.

## Dữ liệu
Dữ liệu huấn luyện và test trong file expert_data.csv. Các feature gồm có:
- education_level: Trình độ học vấn (1: PhD, 2: Master, 3: Bachelor)
- years_of_experience: Số năm kinh nghiệm
- projects: Số dự án đã tham gia
- awards: Số giải thưởng (có thể thay bằng số chứng chỉ)
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


## Hướng dẫn cài đặt và sử dụng
### Cài đặt
Để cài đặt các thư viện cần thiết cho dự án này, chúng ta có thể sử dụng pip. Chạy các lệnh sau để cài đặt các thư viện:
```
pip install pandas
pip install xgboost
pip install tensorflow
pip install scikit-learn
```
### Chạy
Để chạy sử dụng các file tương ứng
```
python test_gradient_boosting.py 
python test_xboost.py
python test_tensorflow.py
python test_rnn.py
```