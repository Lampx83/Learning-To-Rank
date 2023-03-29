# Learning to Rank Project
(Learning to Rank) với 3 thuật toán xếp hạng Gradient Boosting, Tensorflow và XGBoost

Sử dụng 3 thuật toán xếp hạng khác nhau: Gradient Boosting, Tensorflow và XGBoost để xếp hạng dữ liệu.

## Giới thiệu về Learning to Rank
Học để xếp hạng là một lĩnh vực trong Học máy (Machine Learning) và Tìm kiếm thông tin (Information Retrieval) nghiên cứu các phương pháp để xếp hạng các tài liệu dựa trên một số yếu tố nhất định. Các yếu tố này có thể bao gồm từ khóa, tần suất xuất hiện của từ khóa trong tài liệu, sự tương đồng giữa tài liệu và yêu cầu của người dùng, và nhiều yếu tố khác.

## Giới thiệu về các thuật toán xếp hạng

### Gradient Boosting
Gradient Boosting là một thuật toán tập hợp các mô hình dự đoán đơn giản, chẳng hạn như cây quyết định, để tạo ra một mô hình dự đoán mạnh mẽ hơn. Thuật toán này tập trung vào việc tối thiểu hóa sai số của mô hình trên tập huấn luyện bằng cách thêm các mô hình mới và cập nhật trọng số các mẫu.

### XGBoost
XGBoost là một thuật toán Học máy đang được sử dụng rộng rãi trong cả Học sâu và Học máy cổ điển. Nó là một loại thuật toán GBDT (Gradient Boosting Decision Tree) được tối ưu hóa để đạt được hiệu quả cao và thời gian chạy nhanh.

### Tensorflow
Tensorflow là một thư viện Học sâu (Deep Learning) được sử dụng rộng rãi trong nghiên cứu và ứng dụng thực tế. Nó cho phép người dùng xây dựng các mô hình Học sâu như mạng nơ-ron và sử dụng chúng để giải quyết các bài toán phức tạp.

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