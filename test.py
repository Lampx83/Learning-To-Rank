import re

text = "Nguyễn Văn Đức là một giáo sư ngành xây dựng và giảng dạy tại đại học Xây dựng Hà Nội. Trong suốt sự nghiệp của tôi, tôi đã tham gia vào nhiều dự án xây dựng lớn, từ các công trình cầu đường cho đến nhà máy. Giáo sư đã có 100 bài báo và 4 giải thưởng"

pattern = r"\b(giáo sư|tiến sỹ|thạc sỹ)\b"

match = re.search(pattern, text)

if match:
    print(match.group())
else:
    print("Not found")
