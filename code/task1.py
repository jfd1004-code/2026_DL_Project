
from pathlib import Path
import pandas as pd

# 1. 프로젝트 폴더 위치
project_dir = Path(__file__).resolve().parent.parent

# 2. Task 1 데이터 폴더
data_dir = project_dir / "data" / "T1_data"

# 3. CSV 파일 찾기
csv_files = list(data_dir.rglob("*.csv"))

print("발견된 CSV 파일:")
for path in csv_files:
    print(path.name)

# 4. 정답 컬럼이 있는 학습 데이터 찾기
train_file = None

for path in csv_files:
    columns = pd.read_csv(path, nrows=0).columns

    if "is_at_risk" in columns:
        train_file = path
        break

if train_file is None:
    raise FileNotFoundError(
        "is_at_risk 컬럼이 있는 CSV 파일을 찾지 못했습니다."
    )

# 5. 데이터 읽기
df = pd.read_csv(train_file)

print("\n데이터 크기:", df.shape)
print("\n컬럼 이름:")
print(df.columns.tolist())

print("\n상위 5개 행:")
print(df.head())


# Label의 개수 확인
print(df["is_at_risk"].value_counts())

# Label의 비율 확인
print(df["is_at_risk"].value_counts(normalize=True))

# 전체 데이터 정보
print(df.info())

# 결측값 개수 확인
print(df.isna().sum())

print("========== TEST ==========")
print("데이터 크기:", df.shape)
print("정답 분포:")
print(df["is_at_risk"].value_counts())