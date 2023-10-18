import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from torch.optim.lr_scheduler import StepLR

class ClassifierModel(nn.Module):
    def __init__(self, bert_model_name):
        super(ClassifierModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, 5)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

# 예시 데이터
data = [
    ("00년생으로, 포비를 닮았다", "노용훈"),
    ("아무것도 모르는 감자인척하지만 상당한 실력을 갖고 있는 은둔고수이다.", "노용훈"),
    ("말을 안 듣는 교탁 컴퓨터를 수리", "노용훈"),
    ("커피 나눔", "노용훈"),
    ("과제 쳐내는 속도가 매우 빨르다", "노용훈"),
    ("어 나 이거 예전에 제출해서 잘 모르겠는데", "노용훈"),
    ("예약된 커피만 사십개", "노용훈"),
    ("신데렐라", "신용훈"),
    ("잔디 메시지로 악랄한 장난을 친 전적이 있다.", "신용훈"),
    ("통학러", "신용훈"),
    ("하트시그널3에 나오는 정의동과 굉장히 닮았다.", "신용훈"),
    ("스스로를 하남자라고 소개하는데 그 이유는 바로 하남에 살기때문이다.", "신용훈"),
    ("인플루언서", "권정태"),
    ("2020년 1월 14일 5사단 육군으로 입대했다.", "권정태"),
    ("배우 이도현을 닮았다", "권정태"),
    ("현재는 도사님으로 살아가고있다", "권정태"),
    ("반장", "권정태"),
    ("힙한 바지를 입고 다닌다", "권정태"),
    ("교수님처럼 농담을 진담처럼 하는 특징이 있다", "권정태"),
    ("합법적 디도스 공격을 매우 잘한다.", "권정태"),
    ("2023년 8월 8일, 멘토님께 연예인이냐는 소리를 들었다. 역시 BOB 최고의 얼굴, 패션, 두뇌의 소유자다.", "권정태"),
    ("아부를 매우 잘 하는 것으로 알고있다. 무언가 좋은 것을 가지고 있다면 꼭 자랑해보길 바란다. 하지만 말은 끝까지 봐야한다고 끝부터 읽고 이상함을 느낀다면 전송을 멈추길 바란다. ", "권정태"),
    ("7월 15일 15시 30분 이동연에게 영타 타자 대결에서 대패하였다", "권정태"),
    ("장잼민", "장어진"),
    ("초코를 무지 좋아한다.", "장어진"),
    ("근데이제 흰 옷에 묻히면서 먹는걸.", "장어진"),
    ("타격감이 아주 좋다", "장어진"),
    ("오늘은 머리 손질 몇분 걸렸냐고 물어보자.", "장어진"),
    ("외계인 지토를 매우 많이 닮았다.", "장어진"),
    ("3초 안으로 누구든지 꼬실 수 있다고 한다 그의 눈빛이 이상하다면 플러팅 중이다", "장어진"),
    ("푸하하 코드 왜 이렇게 쓰레기 같아?", "장어진"),
    ("통학을 '그딴 거'라고하는 망언을 뱉어 많은 사람들에게 상처를 주었다", "장어진"),
    ("99대장 박민진", "박민진"),
    ("경상남도 비경 100선으로 선정된 삼천포대교가 있는 경상남도 사천시 출신이다.", "박민진"),
    ("서울여대 정보보호학과를 졸업하고 모의 해킹 하다가 하기 싫어서 개발로 커리어 전환을 위해 보안트랙으로 지원하였다.", "박민진"),
    ("핑크를 좋아한다.", "박민진"),
    ("영화 엽문을 본 적이 없다.", "박민진"),
    ("포견자단이랑 마이크 타이슨이 누군지 모른다", "박민진"),
    ("합기도 2단이며 쌍절곤을 잘 한다고 주장하고 있다.", "박민진"),
    ("인스타에 민감하다.", "박민진"),
    ("처음 만난 분과 말하면서 이름은 안 말하면서 러블리라고 자기소개했다는 썰이 있다", "박민진"),
    ("그녀는 사과게임의 초초초고수이다. 무려 159점을 달성했다!", "박민진"),
    ("세계 고양이의 날에 고양이 티셔츠를 입고 왔다.", "박민진"),
    ("쿠로미", "박민진"),
]

model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

def preprocess_data(dataset):
    label_mapping = {"노용훈": 0, "신용훈": 1, "권정태": 2, "장어진": 3, "박민진": 4}
    texts, labels = zip(*dataset)
    encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    encoded_labels = torch.tensor([label_mapping[label] for label in labels])
    return encoded_texts, encoded_labels

train_inputs, train_labels = preprocess_data(data)

# 모델 및 하이퍼파라미터 설정
model = ClassifierModel(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=2, gamma=0.95)  # 스케줄러 추가

# 배치 크기 및 에폭 수 조정
batch_size = 4
epochs = 10

train_data_loader = DataLoader(list(zip(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)), batch_size=batch_size, shuffle=True)

# 모델 학습
model.train()
for epoch in range(epochs):
    for batch in train_data_loader:
        input_ids, attention_mask, labels = [item.to(device) for item in batch]
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    # 스케줄러 스텝 업데이트
    scheduler.step()

    print(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item()}')

# 평가 함수
def evaluate_input(input_text):
    label_mapping = {0: "노용훈", 1: "신용훈", 2: "권정태", 3: "장어진", 4: "박민진"}
    encoded_text = tokenizer.encode_plus(input_text, add_special_tokens=True, return_tensors='pt')
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_class = torch.argmax(logits, dim=1).item()

    return label_mapping[predicted_class]
