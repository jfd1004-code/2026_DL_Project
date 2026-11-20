
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

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


# 1. 입력 Feature만 선택
X = df.drop(columns=["sample_id", "is_at_risk"])

# 2. 정답 Label 선택
y = df["is_at_risk"]

print("\n=== Feature 개수 ===")
print(X.shape)

print("\n=== Feature 이름 ===")
print(X.columns.tolist())

print("\n=== Feature 데이터 타입 ===")
print(X.dtypes)

# 3. 특수값 개수 확인
print("\n=== 특수값 개수 ===")

for value in [-7, -8, -9]:
    count = (X == value).sum().sum()
    print(f"{value}: {count}개")


# Feature별 특수값 개수 분석

special_counts = pd.DataFrame({
    "-7": (X == -7).sum(),
    "-8": (X == -8).sum(),
    "-9": (X == -9).sum()
})

# 각 Feature의 특수값 총합
special_counts["total"] = special_counts.sum(axis=1)

# 특수값이 있는 Feature만 표시
result = special_counts[
    special_counts["total"] > 0
]

# 특수값이 많은 순서로 정렬
result = result.sort_values(
    by="total",
    ascending=False
)

print("\n=== Feature별 특수값 분포 ===")
print(result.to_string())


from sklearn.model_selection import train_test_split

# 입력 데이터와 정답 데이터 분리
X = df.drop(columns=["sample_id", "is_at_risk"])
y = df["is_at_risk"]

# Training / Validation 분리
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n=== Train / Validation Split ===")

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)

print("\nTraining Label 분포:")
print(y_train.value_counts(normalize=True))

print("\nValidation Label 분포:")
print(y_val.value_counts(normalize=True))


import numpy as np
from sklearn.preprocessing import StandardScaler

# 원본 데이터 보존
X_train_p = X_train.copy()
X_val_p = X_val.copy()

# 기존 숫자 Feature 목록
numeric_cols = X_train.columns.tolist()

# 1. 특수값별 Indicator 생성
special_codes = [-7, -8, -9]

for col in numeric_cols:
    for code in special_codes:
        if (X_train[col] == code).any():
            name = f"{col}_is_{abs(code)}"

            X_train_p[name] = (X_train[col] == code).astype(int)
            X_val_p[name] = (X_val[col] == code).astype(int)

# 2. 특수값을 NaN으로 변경
X_train_p[numeric_cols] = (
    X_train_p[numeric_cols]
    .astype(float)
    .replace(special_codes, np.nan)
)

X_val_p[numeric_cols] = (
    X_val_p[numeric_cols]
    .astype(float)
    .replace(special_codes, np.nan)
)

# 3. Training 데이터의 중앙값 계산
medians = X_train_p[numeric_cols].median()

# 같은 중앙값을 Train / Validation에 적용
X_train_p[numeric_cols] = X_train_p[numeric_cols].fillna(medians)
X_val_p[numeric_cols] = X_val_p[numeric_cols].fillna(medians)

# 4. 정규화 (Training 데이터로만 fit)
scaler = StandardScaler()

X_train_p[numeric_cols] = scaler.fit_transform(
    X_train_p[numeric_cols]
)

X_val_p[numeric_cols] = scaler.transform(
    X_val_p[numeric_cols]
)

# 5. 결과 확인
print("\n=== 전처리 결과 ===")

print("X_train:", X_train_p.shape)
print("X_val:", X_val_p.shape)

print("추가된 Indicator 개수:",
      X_train_p.shape[1] - len(numeric_cols))

print("Training NaN:", X_train_p.isna().sum().sum())
print("Validation NaN:", X_val_p.isna().sum().sum())

import torch
from torch.utils.data import TensorDataset, DataLoader

# 1. Pandas DataFrame을 Tensor로 변환
X_train_tensor = torch.from_numpy(
    X_train_p.to_numpy(dtype="float32")
)

X_val_tensor = torch.from_numpy(
    X_val_p.to_numpy(dtype="float32")
)

# 2. Label도 Tensor로 변환
y_train_tensor = torch.tensor(
    y_train.to_numpy(),
    dtype=torch.float32
).reshape(-1, 1)

y_val_tensor = torch.tensor(
    y_val.to_numpy(),
    dtype=torch.float32
).reshape(-1, 1)

# 3. 입력 데이터와 정답을 하나로 묶기
train_dataset = TensorDataset(
    X_train_tensor, y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor, y_val_tensor
)

# 4. Mini-batch를 생성하는 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

# 5. 결과 확인
print("\n=== DataLoader 결과 ===")

print("Train Tensor:", X_train_tensor.shape)
print("Validation Tensor:", X_val_tensor.shape)
print("Train Label:", y_train_tensor.shape)

print("Training Batch 개수:", len(train_loader))
print("Validation Batch 개수:", len(val_loader))

# 첫 번째 Batch 확인
batch_X, batch_y = next(iter(train_loader))

print("Batch X Shape:", batch_X.shape)
print("Batch y Shape:", batch_y.shape)


import torch.nn as nn

# GPU 사용 가능하면 GPU, 아니면 CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# DNN 모델 생성
model = nn.Sequential(
    nn.Linear(35, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# 모델을 연산 장치로 이동
model = model.to(device)

print("\n=== DNN Model ===")
print(model)
print("Device:", device)


# 첫 번째 Mini-batch를 모델에 전달
batch_X = batch_X.to(device)

# 학습하지 않고 Forward Propagation만 실행
model.eval()

with torch.no_grad():
    logits = model(batch_X)
    probabilities = torch.sigmoid(logits)

print("\n=== Forward Propagation ===")

print("Input Shape:", batch_X.shape)
print("Output Shape:", logits.shape)

print("\n예측 확률 5개:")
print(probabilities[:5])


import torch.optim as optim

# 1. Loss Function
criterion = nn.BCEWithLogitsLoss()

# 2. Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# 3. Training

# 최대 학습 횟수
epochs = 50

# 7 Epoch 연속 개선이 없으면 종료
patience = 7
epochs_no_improve = 0

# 학습 결과 저장
train_losses = []
val_losses = []
val_aucs = []

# 최고 성능 기록
best_auc = -float("inf")
best_epoch = 0

# 최고 성능 모델 저장 경로
checkpoint_path = project_dir / "task1.pt"


for epoch in range(epochs):

    # 1. Training
    model.train()
    train_loss_sum = 0.0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * batch_X.size(0)

    train_loss = train_loss_sum / len(train_dataset)

    # 2. Validation
    model.eval()

    val_loss_sum = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            probs = torch.sigmoid(logits)

            val_loss_sum += loss.item() * batch_X.size(0)

            all_probs.extend(
                probs.cpu().numpy().ravel()
            )
            all_labels.extend(
                batch_y.cpu().numpy().ravel()
            )

    val_loss = val_loss_sum / len(val_dataset)
    val_auc = roc_auc_score(all_labels, all_probs)

    # 3. 결과 기록
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_aucs.append(val_auc)


# 최고 AUROC를 갱신하면 Weight 저장

    # 최고 AUROC 갱신 확인
    if val_auc > best_auc:
        best_auc = val_auc
        best_epoch = epoch + 1

        # 개선됐으므로 카운터 초기화
        epochs_no_improve = 0

        # 최고 모델 저장
        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(
            f"Best model saved! "
            f"Epoch: {best_epoch}, "
            f"AUROC: {best_auc:.4f}"
        )

    else:
        # 개선되지 않았으면 1 증가
        epochs_no_improve += 1

    # Early Stopping 조건 확인
    if epochs_no_improve >= patience:
        print(f"\nEarly Stopping at Epoch {epoch + 1}")
        break


    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val AUROC: {val_auc:.4f}"
    )

print("\nBest Validation AUROC:", max(val_aucs))
print("Best Epoch:", val_aucs.index(max(val_aucs)) + 1)



from sklearn.metrics import roc_auc_score

# 평가 모드로 전환
model.eval()

all_probs = []
all_labels = []

with torch.no_grad():

    for batch_X, batch_y in val_loader:

        batch_X = batch_X.to(device)

        # 예측값 계산
        logits = model(batch_X)

        # Logit을 확률로 변환
        probs = torch.sigmoid(logits)

        # CPU로 옮겨 결과 저장
        all_probs.extend(
            probs.cpu().numpy().ravel()
        )

        all_labels.extend(
            batch_y.numpy().ravel()
        )

# Validation AUROC 계산
val_auc = roc_auc_score(
    all_labels,
    all_probs
)

print("\n=== Validation Result ===")
print(f"AUROC: {val_auc:.4f}")


print("\n=== Best Model ===")
print("Best Epoch:", best_epoch)
print(f"Best AUROC: {best_auc:.4f}")

# 최고 성능 모델의 Weight 복원
model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Best model loaded successfully!")
